import torch
from torch import nn


class HyperbolicGraphConvolution(nn.Module):
    """LLGCN local and global Lorentz centroids (Eqs. 4–6)."""

    def __init__(self, manifold, num_layers):
        super().__init__()
        self.manifold = manifold
        self.num_layers = num_layers

    def centroid(self, x):
        norm = (-self.manifold.inner(x, x, keepdim=True)).clamp_min(1e-8).sqrt()
        return self.manifold.k.sqrt() * x / norm

    def forward(self, x, adj):
        layers = []
        for _ in range(self.num_layers):
            x = self.centroid(torch.sparse.mm(adj, x))
            layers.append(x)
        return x if len(layers) == 1 else self.centroid(torch.stack(layers).sum(0))
