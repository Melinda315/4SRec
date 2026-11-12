import math

import torch
from geoopt import Lorentz, ManifoldParameter
from torch import nn
from torch.nn import functional as F

from .hyp_layers import HyperbolicGraphConvolution


class TaxoRec(nn.Module):
    """4SRec's trainable semantic and structural views on the unit hyperboloid."""

    def __init__(self, semantic, num_users, args):
        super().__init__()
        self.manifold = Lorentz(k=1.0).to(semantic.device)
        self.encoder = HyperbolicGraphConvolution(self.manifold, args.num_layers)
        self.num_users = num_users
        self.margin = args.margin
        self.beta = args.beta
        self.temperature = args.temperature
        self.id_embedding = self.initialize(torch.randn_like(semantic) * args.scale)
        self.text_embedding = self.initialize(semantic)

    def initialize(self, x):
        with torch.no_grad():
            points = self.manifold.expmap0(x)
        return ManifoldParameter(points, manifold=self.manifold)

    def encode(self, adj):
        text = self.manifold.projx(self.text_embedding)
        structural = self.encoder(self.manifold.projx(self.id_embedding), adj)
        return text, structural

    def distance_from_inner(self, inner):
        # Stay inside acosh's differentiable domain despite float32 roundoff.
        return torch.acosh((-inner).clamp_min(1.0 + torch.finfo(inner.dtype).eps))

    def distance(self, x, y):
        return self.distance_from_inner(self.manifold.inner(x, x, y))

    def distance_matrix(self, x, y):
        inner = x[:, 1:] @ y[:, 1:].T - x[:, :1] @ y[:, :1].T
        return self.distance_from_inner(inner)

    def user_weights(self, structural, users):
        radius = self.distance_from_inner(-structural[:self.num_users, 0])
        radius = radius[users].clamp_max(math.sqrt(15))
        scaled = (radius - radius.min()) / (radius.max() - radius.min()).clamp_min(1e-8)
        return 0.5 + 0.5 * scaled

    def contrastive_loss(self, text, structural):
        cross = -self.distance_matrix(text, structural).square()
        same_text = -self.distance_matrix(text, text).square()
        same_id = -self.distance_matrix(structural, structural).square()
        positive = cross.diagonal() / self.temperature
        diagonal = torch.eye(len(text), dtype=torch.bool, device=text.device)
        # Legacy negatives retain diagonal exp(0) terms and divide after exponentiation.
        negatives = torch.cat(
            [scores.masked_fill(diagonal, 0) for scores in (cross, same_text, same_id)], dim=1
        ) - math.log(self.temperature)
        logits = torch.cat((positive[:, None], negatives), dim=1)
        return (torch.logsumexp(logits, dim=1) - positive).mean()

    def compute_loss(self, text, structural, triples):
        users, positives, negatives = triples.T
        def hinge(view):
            positive = self.distance(view[users], view[positives]).square().clamp_max(50)
            negative = self.distance(view[users], view[negatives]).square().clamp_max(50)
            return F.relu(self.margin + positive - negative)

        text_loss = hinge(text).sum()
        id_loss = (self.user_weights(structural, users) * hinge(structural)).sum()
        services = positives.unique()
        contrastive = self.contrastive_loss(text[services], structural[services])
        return text_loss + id_loss + self.beta * contrastive

    def predict(self, text, structural, users):
        text_dist = self.distance_matrix(text[users], text[self.num_users:])
        id_dist = self.distance_matrix(structural[users], structural[self.num_users:])
        return -(text_dist.square() + id_dist.square())
