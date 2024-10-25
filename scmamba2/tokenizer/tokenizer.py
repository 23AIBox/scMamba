import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class scEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(
            self, 
            feature_size,
            patch_size,
            d_model, 
            hidden_dropout_prob=0.1,
            use_mask_token: bool = False
            ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, d_model))
            if use_mask_token
            else None
        )
        
        self.patch_embeddings = PatchEmbeddings(
            feature_size=feature_size, patch_size=patch_size, d_model=d_model
        )

        num_patches = self.patch_embeddings.num_patches

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches, d_model)
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.num_patches = num_patches
        self.patch_size = patch_size

    def forward(
        self,
        seq_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:

        batch_size = seq_values.shape[0]
        
        embeddings = self.patch_embeddings(
            seq_values
        )
        # embeddings.size() = [batch_size, num_patches, emded_dim]

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # embeddings.size() = [batch_size, num_patches + 1, emded_dim]

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        # embeddings = self.dropout(embeddings)
        return embeddings  # [batch_size, num_patches + 1, emded_dim]


class PatchEmbeddings(nn.Module):
    """
    rna and atac sequencing to Patch Embedding.

    """

    def __init__(
            self,
            feature_size,
            patch_size,
            d_model
            ):
        super().__init__()
        embed_dim = d_model
        num_patches = math.ceil(
            feature_size / patch_size
        )
        pad_size = num_patches * patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        # x.size() = [batch_size, seq_length]
        x = F.pad(x, (0, self.pad_size)).view(
            x.shape[0], self.num_patches, self.patch_size
        ) # [batch_size, num_patches, pathc_size]
        x = self.projection(x) # [batch_size, num_patches, emded_dim]
        return x


