import torch
import torch.nn as nn
import torch.nn.functional as F 

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    atac_loss = contrastive_loss(similarity.T)
    return (caption_loss + atac_loss) / 2.0


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=2.6592, cos_simi_scale=1.0, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * logit_scale, requires_grad=requires_grad
        )
        self.cos_simi_scale = cos_simi_scale
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, rna_embeds, atac_embeds):

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_atac = torch.matmul(atac_embeds, rna_embeds.t()) * logit_scale

        loss = clip_loss(logits_per_atac)
        # cosin_similarity = F.cosine_similarity(rna_embeds, atac_embeds)
        cosin_similarity_loss = 1 - torch.mean(self.cos(rna_embeds, atac_embeds))
        # cosin_similarity_loss = cosin_similarity_loss * logit_scale
        loss += (cosin_similarity_loss * self.cos_simi_scale)

        return loss, logits_per_atac
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.2, cos_simi_scale=1):
        """
        Implements the contrastive loss function with learnable temperature parameter.

        Args:
            initial_temperature: Scalar, the initial value for the temperature \u03c3.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos_simi_scale = cos_simi_scale
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, x, y):
        """
        Computes the contrastive loss.

        Args:
            x: Tensor of shape (N, D), where N is the batch size and D is the feature dimension (image embeddings).
            y: Tensor of shape (N, D), where N is the batch size and D is the feature dimension (text embeddings).

        Returns:
            loss: Scalar, the computed contrastive loss.
        """
        # Compute similarity matrices
        sim_matrix = torch.matmul(x, y.T) / self.temperature  # Shape: (N, N)

        # Compute image-to-text loss
        image_to_text_loss = -torch.mean(
            torch.log_softmax(sim_matrix, dim=1).diag()
        )

        # Compute text-to-image loss
        text_to_image_loss = -torch.mean(
            torch.log_softmax(sim_matrix.T, dim=1).diag()
        )

        # Total contrastive loss
        loss = (image_to_text_loss + text_to_image_loss) / 2
        cosine_similarity_loss = 1 - torch.mean(self.cos(x, y))
        loss += cosine_similarity_loss * self.cos_simi_scale
        return loss, loss