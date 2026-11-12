"""Create ServiceLLM SCFT examples using Algorithm 1 and training histories only."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from model.prompts import item_prompt, load_prompt_inputs


def preferences(train, attributes, minimum, dominance):
    for user, history in sorted(train.items()):
        if len(history) < minimum:
            continue
        counts = Counter(attr for item in history for attr in attributes[item])
        ratio = max(counts.values(), default=0) / len(history)
        if ratio >= dominance:
            dominant = {attr for attr, n in counts.items() if n / len(history) >= dominance}
            yield user, dominant, math.log1p(len(history)) * ratio


def choose_triplets(train, attributes, args):
    by_attribute = defaultdict(set)
    for item, attrs in enumerate(attributes):
        for attr in attrs:
            by_attribute[attr].add(item)
    rng = random.Random(args.seed)
    pool = list(preferences(train, attributes, args.min_history, args.dominance))
    emitted = 0
    while pool and emitted < args.users:
        index = rng.choices(range(len(pool)), weights=[weight for _, _, weight in pool])[0]
        user, dominant, _ = pool.pop(index)
        history = set(train[user])
        positives = [i for i in sorted(history) if attributes[i] & dominant]
        rng.shuffle(positives)
        for pos in positives:
            candidates = set().union(*(by_attribute[attr] for attr in attributes[pos]))
            hard = sorted(i for i in candidates - history if len(attributes[i] & attributes[pos]) == 1)
            if len(hard) >= args.negatives:
                emitted += 1
                yield user, pos, rng.sample(hard, args.negatives)
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--dominance", type=float, default=0.3)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.min_history, args.users, args.negatives) < 1 or not 0 < args.dominance <= 1:
        parser.error("Counts must be positive and dominance must be in (0, 1]")

    train, _, items = load_prompt_inputs(args.input_dir)
    attributes = [
        {(key, item[key]) for key in ("item_category", "item_brand") if item.get(key)}
        for item in items
    ]
    anchors = list(choose_triplets(train, attributes, args))
    if not anchors:
        raise ValueError("No feasible anchors with dominant positives and enough hard negatives")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for user, positive, negatives in anchors:
            lines = [item_prompt(items[i]) + "." for i in train[user] if i != positive]
            text = "Previously interacted items:\n" + "".join(
                f"{j}. {line}\n" for j, line in enumerate(lines, 1)
            )
            for item, label in [(positive, "pos"), *((i, "neg") for i in negatives)]:
                row = {"user_id": user, "user_text": text, "item": item_prompt(items[item]) + ".", "label": label}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"sampled_anchors={len(anchors)} requested_anchors={args.users} "
          f"triplets={len(anchors) * args.negatives}")


if __name__ == "__main__":
    main()
