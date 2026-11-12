import argparse
import json
import random
from pathlib import Path

import torch
from geoopt.optim import RiemannianSGD

from model.data import load_inputs, train_graph
from model.eval_metrics import evaluate
from model.model import TaxoRec


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate 4SRec (LLGCN + InteRec).")
    parser.add_argument("--dataset", default="Scientific")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--embedding", type=Path, help="Override the PCA tensor path")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for checkpoint and metrics")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum training epochs")
    parser.add_argument("--eval-freq", type=int, default=20, help="Epochs between evaluations")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.75, help="Contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    if min(args.epochs, args.eval_freq, args.batch_size, args.num_layers) < 1 or args.temperature <= 0:
        parser.error("epochs, eval-freq, batch-size, num-layers and temperature must be positive")
    return args


def batches(edges, train, num_items, batch_size, rng, device):
    rng.shuffle(edges)
    num_users = len(train)
    for start in range(0, len(edges), batch_size):
        triples = []
        for user, positive in edges[start:start + batch_size]:
            negative = rng.randrange(num_items)
            while negative in train[user]:
                negative = rng.randrange(num_items)
            triples.append((user, positive, negative + num_users))
        yield torch.tensor(triples, dtype=torch.long, device=device)


def main():
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train, valid, test, num_items, semantic = load_inputs(args.data_root, args.dataset, args.embedding)
    edges, adj = train_graph(train, num_items, device)
    model = TaxoRec(semantic.to(device), len(train), args)
    optimizer = RiemannianSGD(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
    )

    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "model.pt"
    selection_metrics = None
    best_recall = {10: 0.0, 20: 0.0}
    stale_evaluations = 0
    selection_split = "validation" if valid else "test"
    selection_targets = valid or test
    print(f"device={device} users={len(train)} items={num_items} train_edges={len(edges)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for triples in batches(edges, train, num_items, args.batch_size, rng, device):
            optimizer.zero_grad()
            loss = model.compute_loss(*model.encode(adj), triples)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
        print(f"epoch={epoch} train_loss={total:.6f}")

        # A run shorter than the evaluation interval still needs a checkpoint.
        evaluate_at_end = epoch == args.epochs and (valid or selection_metrics is None)
        if epoch % args.eval_freq != 0 and not evaluate_at_end:
            continue

        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, model.encode(adj), selection_targets, train)
        print(f"{selection_split} epoch={epoch}", json.dumps(metrics))

        if valid:
            improved = selection_metrics is None or metrics["Recall@10"] > selection_metrics["Recall@10"]
        else:
            improved_cutoffs = set()
            for k in (10, 20):
                recall = metrics[f"Recall@{k}"]
                if recall > best_recall[k]:
                    best_recall[k] = recall
                    improved_cutoffs.add(k)
            improved = bool(improved_cutoffs)
            # Preserve the original order: Recall@10 resets before the Recall@20 check.
            if 10 in improved_cutoffs:
                stale_evaluations = 0
            stale_evaluations = 0 if 20 in improved_cutoffs else stale_evaluations + 1

        if improved or selection_metrics is None:
            selection_metrics = metrics
            selected_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        if stale_evaluations == 5:
            print(f"early_stop epoch={epoch} selected_epoch={selected_epoch}")
            break

    if valid:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, model.encode(adj), test, train, valid)
    else:
        metrics = selection_metrics
    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "torch": str(torch.__version__),
        "selection_split": selection_split,
        "selected_epoch": selected_epoch,
        "stopped_epoch": epoch,
        "validation": selection_metrics if valid else None,
        "test": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"selected_epoch={selected_epoch}")
    print("test", json.dumps(metrics))


if __name__ == "__main__":
    main()
