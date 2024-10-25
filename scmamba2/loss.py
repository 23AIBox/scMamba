import torch
import torch.nn as nn


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    atac_loss = contrastive_loss(similarity.T)
    return (caption_loss + atac_loss) / 2.0


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=2.6592, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * logit_scale, requires_grad=requires_grad
        )

    def forward(self, rna_embeds, atac_embeds):
        # normalized features
        # atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        # rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_atac = torch.matmul(atac_embeds, rna_embeds.t()) * logit_scale

        loss = clip_loss(logits_per_atac)

        return loss, logits_per_atac