import torch
import torch.nn.functional as F



def user_pos_neg_infonce_loss(
    user_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    neg_embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
    pos_embeddings = F.normalize(pos_embeddings, p=2, dim=-1)
    neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)
    pos_sim = (user_embeddings * pos_embeddings).sum(dim=-1) / temperature
    neg_sim = (user_embeddings * neg_embeddings).sum(dim=-1) / temperature
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss
