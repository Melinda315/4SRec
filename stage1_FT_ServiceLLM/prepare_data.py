"""Convert Amazon 2018 or Yelp raw files to train/test inputs and ordered prompt metadata."""

import argparse
import csv
import gzip
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def json_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_amazon(interactions, metadata):
    # Keep one event per implicit user–item pair, using the latest timestamp.
    pairs = {}
    with interactions.open(encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if not row or row[0].lower() in {"item", "item_id", "asin"}:
                continue
            item, user, _, timestamp = row
            key = (user, item)
            pairs[key] = max(int(timestamp), pairs.get(key, 0))
    wanted = {item for _, item in pairs}
    items = {}
    for row in json_lines(metadata):
        item = row.get("asin")
        if item in wanted and item not in items:
            category = row.get("category") or []
            items[item] = {
                "item": item,
                "item_name": str(row.get("title") or item),
                "item_category": " > ".join(map(str, category)) if isinstance(category, list) else str(category),
                "item_brand": str(row.get("brand") or "").strip(),
            }
    return pairs, items


def load_yelp(interactions, metadata):
    pairs = {}
    for row in json_lines(interactions):
        if row["stars"] >= 3:
            key = (row["user_id"], row["business_id"])
            pairs[key] = max(row["date"], pairs.get(key, ""))
    wanted = {item for _, item in pairs}
    items = {
        row["business_id"]: {
            "item": row["business_id"],
            "item_name": row["name"],
            "item_category": row.get("categories") or "",
        }
        for row in json_lines(metadata) if row["business_id"] in wanted
    }
    return pairs, items


def filter_core(pairs, k):
    while pairs:
        users = Counter(u for u, _ in pairs)
        items = Counter(i for _, i in pairs)
        filtered = {p: t for p, t in pairs.items() if users[p[0]] >= k and items[p[1]] >= k}
        if len(filtered) == len(pairs):
            return filtered
        pairs = filtered
    raise ValueError("No interactions remain after filtering")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["Scientific", "Arts", "Office", "yelp"], required=True)
    parser.add_argument("--interactions", type=Path, required=True, help="Ratings CSV or Yelp review JSONL")
    parser.add_argument("--metadata", type=Path, required=True, help="Amazon metadata JSONL.gz or Yelp business JSONL")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--k-core", type=int, help="Default: Amazon 5, Yelp 10")
    args = parser.parse_args()
    k = args.k_core if args.k_core is not None else (10 if args.dataset == "yelp" else 5)
    if k < 2:
        parser.error("--k-core must be at least 2 to hold out a target per user")
    loader = load_yelp if args.dataset == "yelp" else load_amazon
    pairs, metadata = loader(args.interactions, args.metadata)
    pairs = filter_core({p: t for p, t in pairs.items() if p[1] in metadata}, k)
    users = sorted({u for u, _ in pairs})
    items = sorted({i for _, i in pairs})
    item_index = {item: index for index, item in enumerate(items)}
    histories = defaultdict(list)
    for (user, item), time in pairs.items():
        histories[user].append((time, item_index[item]))

    train, test = {}, {}
    for u, user in enumerate(users):
        history = [i for _, i in sorted(histories[user])]
        n_train = 4 * len(history) // 5
        train[u], test[u] = history[:n_train], history[n_train:]

    destination = args.data_root / args.dataset
    destination.mkdir(parents=True)
    for name, split in (("train", train), ("test", test)):
        with (destination / f"{name}.pkl").open("wb") as file:
            pickle.dump(split, file, protocol=4)
    write_jsonl(destination / "users.jsonl", ({"user": u} for u in users))
    write_jsonl(destination / "items.jsonl", (metadata[i] for i in items))
    print(f"users={len(users)} items={len(items)} "
          f"train={sum(map(len, train.values()))} test={sum(map(len, test.values()))}")


if __name__ == "__main__":
    main()
