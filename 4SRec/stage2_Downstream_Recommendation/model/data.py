import pickle
from pathlib import Path

import torch


def read_split(path):
    with path.open("rb") as file:
        rows = pickle.load(file)
    if not isinstance(rows, dict) or any(
        type(user) is not int or user < 0 or not isinstance(items, (list, tuple))
        or any(type(item) is not int or item < 0 for item in items)
        for user, items in rows.items()
    ):
        raise ValueError(f"{path} must contain dict[int, list[int]] with nonnegative IDs")
    return {user: set(items) for user, items in rows.items()}


def load_inputs(data_root, dataset, embedding=None):
    directory = Path(data_root) / dataset
    train = read_split(directory / "train.pkl")
    valid_path = directory / "valid.pkl"
    valid = read_split(valid_path) if valid_path.exists() else {}
    test = read_split(directory / "test.pkl")
    if not train or set(train) != set(range(len(train))) or not any(train.values()):
        raise ValueError("train.pkl must contain every contiguous user ID and at least one training interaction")
    if not test or (valid_path.exists() and not valid):
        raise ValueError("Each supplied evaluation split must contain evaluable users")
    for split in (valid, test):
        if not split.keys() <= train.keys() or not all(split.values()):
            raise ValueError("Evaluation users must exist in train and have nonempty targets")
    for user, items in train.items():
        valid_items = valid.get(user, set())
        test_items = test.get(user, set())
        if items & (valid_items | test_items) or valid_items & test_items:
            raise ValueError(f"Split overlap for user {user}")
    items = set().union(*train.values(), *valid.values(), *test.values())
    if items != set(range(len(items))) or len(items) < 2:
        raise ValueError("Item IDs must be contiguous, with at least two items")
    if any(len(history) == len(items) for history in train.values()):
        raise ValueError("Every user must have at least one item available for negative sampling")

    embedding_path = embedding or Path(data_root) / "emb" / dataset / f"{dataset.lower()}.pt"
    semantic = torch.load(embedding_path, map_location="cpu", weights_only=True)
    expected_rows = len(train) + len(items)
    if (
        not isinstance(semantic, torch.Tensor) or semantic.ndim != 2
        or semantic.shape[0] != expected_rows or not semantic.shape[1]
        or not semantic.is_floating_point() or not torch.isfinite(semantic).all()
    ):
        raise ValueError(f"{embedding_path} must be a finite tensor with {expected_rows} rows")
    return train, valid, test, len(items), semantic.float()


def train_graph(train, num_items, device):
    num_users = len(train)
    edges = [(u, num_users + i) for u in sorted(train) for i in sorted(train[u])]
    pairs = torch.tensor(edges, dtype=torch.long)
    n = num_users + num_items
    nodes = torch.arange(n)
    indices = torch.cat((pairs.T, pairs.flip(1).T, torch.stack((nodes, nodes))), dim=1)
    weights = torch.bincount(indices[0], minlength=n).float().reciprocal()[indices[0]]
    adj = torch.sparse_coo_tensor(indices, weights, (n, n)).coalesce().to(device)
    return edges, adj
