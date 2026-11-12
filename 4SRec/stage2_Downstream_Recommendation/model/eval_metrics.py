import math

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, views, targets, train, valid=None, batch_size=128):
    """Macro-average exact full-catalog Recall and binary NDCG (Section 4.1.1)."""
    cutoffs = (5, 10, 20, 50)
    totals = {f"{metric}@{k}": 0.0 for k in cutoffs for metric in ("Recall", "NDCG")}
    discounts = 1.0 / np.log2(np.arange(2, max(cutoffs) + 2))
    users = sorted(targets)
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = model.predict(*views, batch).cpu().numpy()
        for user, row in zip(batch, scores):
            seen = train[user] | (valid.get(user, set()) if valid else set())
            row[list(seen)] = -np.inf
            ranks = np.argsort(-row, kind="stable")[:max(cutoffs)]
            hits = [item in targets[user] and math.isfinite(row[item]) for item in ranks]
            for k in cutoffs:
                totals[f"Recall@{k}"] += sum(hits[:k]) / len(targets[user])
                dcg = np.dot(hits[:k], discounts[:min(k, len(hits))])
                totals[f"NDCG@{k}"] += dcg / discounts[:min(k, len(targets[user]))].sum()
    return {name: float(value / len(users)) for name, value in totals.items()}
