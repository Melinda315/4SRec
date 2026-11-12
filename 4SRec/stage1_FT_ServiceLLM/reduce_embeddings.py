"""Apply a joint user/item PCA and the source implementation's L2 normalization."""

import argparse
from pathlib import Path

import torch
from sklearn.decomposition import PCA
from torch.nn.functional import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    tensor = torch.load(args.input, map_location="cpu", weights_only=True)
    if tensor.ndim != 2 or not 1 <= args.components <= min(tensor.shape):
        parser.error("--components must be positive and within both matrix dimensions")
    pca = PCA(n_components=args.components, random_state=args.seed)
    projected = normalize(torch.from_numpy(pca.fit_transform(tensor.float().numpy())), dim=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projected.contiguous(), args.output)
    print(f"pca_embedding_shape={tuple(projected.shape)}")


if __name__ == "__main__":
    main()
