# Copyright (c) 2023, Albert Gu, Tri Dao.
import sys
sys.path.append("..")

import math
from functools import partial
import json
import os
import copy

import scanpy as sc
from tqdm import tqdm
from anndata import AnnData, concat
from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import Subset

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from ..tokenizer.tokenizer import scEmbeddings
from ..utils.plot import plot_umap, plot_paired_umap
from .. import logger

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        feature_size: int,
        patch_size: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.embedding = scEmbeddings(feature_size, patch_size, d_model)
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states
    
class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        d_feature: int,
        patch_size: int,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            feature_size=d_feature,
            patch_size=patch_size,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.active = nn.Tanh()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer= n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        # self.tie_weights()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, pool='last token', normalize=False, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if pool == 'mean':
            hidden_states = hidden_states.mean(dim=1)
        elif pool == 'last token':
            hidden_states = hidden_states[:, -1] 
        # if num_last_tokens > 0:
        #     hidden_states = hidden_states[:, -num_last_tokens:]
        hidden_states = self.active(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        if normalize:
            lm_logits = lm_logits / torch.linalg.norm(lm_logits, ord=2, dim=-1, keepdim=True)
        return lm_logits

    def get_representaion(
            self, 
            dataloader, 
            cell_type="cell_type", 
            modality="rna", 
            out_dir=".", 
            device="cuda",
            n_neighbors=30, 
            min_dist=0.5, 
            metric="cosine", 
            resolution=1
        ):
        if isinstance(dataloader.dataset, Subset):
            obs = dataloader.dataset.dataset.mdata.obs.iloc[dataloader.dataset.indices, :].copy()
        else:
            obs = dataloader.dataset.mdata.obs
        fn = self.forward
        if modality == "rna":
            logger.info("Getting scRNA-seq representation ...")
            adata = torch.concat(
                [
                    fn(rna.to(device).float(), num_last_tokens=1).detach().cpu()
                    for (rna, atac) in tqdm(dataloader, desc=modality)
                ]
            ).numpy()
        else:
            logger.info("Getting scATAC-seq representation ...")
            adata = torch.concat(
                [
                    fn(atac.to(device).float(), num_last_tokens=1).detach().cpu()
                    for (rna, atac) in tqdm(dataloader, desc=modality)
                ]
            ).numpy()
        adata = AnnData(adata, obs=obs)
        sc.settings.figdir = out_dir
        plot_umap(
            adata, 
            color=cell_type, 
            metric=metric, 
            save=f"_{modality}.png",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            resolution=resolution 
        )
        adata.write(f"{out_dir}/{modality}.h5ad")
        
        return adata
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

class scMambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        d_feature_omics1: int,
        d_feature_omics2: int,
        patch_size: int,
        multi_batches: bool=False,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.multi_batches = multi_batches
        super().__init__()

        self.batch_norm_omics1 = nn.BatchNorm1d(num_features=d_feature_omics1)
        self.batch_norm_omics2 = nn.BatchNorm1d(num_features=d_feature_omics2)
        
        self.encoder_omics1 = MambaLMHeadModel(
            config=config, d_feature=d_feature_omics1, patch_size=patch_size, device=device
        )
        
        self.encoder_omics2 = MambaLMHeadModel(
            config=config, d_feature=d_feature_omics2, patch_size=patch_size, device=device
        )
        # Initialize weights and apply final processing
        # self.apply(
        #     partial(
        #         _init_weights,
        #         n_layer=n_layer,
        #         **(initializer_cfg if initializer_cfg is not None else {}),
        #     )
        # )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids_omics_1, input_ids_omics_2, position_ids=None, inference_params=None, num_last_tokens=0, pool='last token', normalize=False, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        input_ids_omics_1 = input_ids_omics_1 if not self.multi_batches \
            else self.batch_norm_omics1(input_ids_omics_1)
        input_ids_omics_2 = input_ids_omics_2 if not self.multi_batches \
            else self.batch_norm_omics2(input_ids_omics_2)
        lm_logits_omics1 = self.encoder_omics1(input_ids_omics_1, inference_params=inference_params, num_last_tokens=num_last_tokens, pool=pool, normalize=normalize, **mixer_kwargs)
        lm_logits_omics2 = self.encoder_omics2(input_ids_omics_2, inference_params=inference_params, num_last_tokens=num_last_tokens, pool=pool, normalize=normalize, **mixer_kwargs)
        # lm_logits_omics1, lm_logits_omics2 = torch.squeeze(lm_logits_omics1, 1), torch.squeeze(lm_logits_omics2, 1)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits_omics_1", "logits_omics_2"])
        return CausalLMOutput(logits_omics_1=lm_logits_omics1, logits_omics_2=lm_logits_omics2)
    
    
    def get_representation(
            self, 
            dataloader, 
            cell_type="cell_type", 
            out_dir=".",  
            device="cuda", 
            n_neighbors=30, 
            min_dist=0.5, 
            metric="cosine", 
            resolution=1
        ):
        if dataloader is not None:
            # get rna embeddings and atac embeddings
            rna_emb = self.encoder_omics1.get_representaion(
                dataloader, cell_type=cell_type, modality="rna", out_dir=out_dir, device=device,
                n_neighbors=n_neighbors, metric=metric, min_dist=min_dist, resolution=resolution
            )
            atac_emb = self.encoder_omics2.get_representaion(
                dataloader, cell_type=cell_type, modality="atac", out_dir=out_dir, device=device,
                n_neighbors=n_neighbors, metric=metric, min_dist=min_dist, resolution=resolution
            )          
            concat_emb = concat(
                [atac_emb, rna_emb],
                label="modality",
                keys=["atac", "rna"],
                index_unique="#",
            )
            concat_emb = concat_emb[concat_emb.obs['cell_type'].notnull()].copy()
            plot_paired_umap(
                    concat_emb,
                    metric=metric,
                    color=['modality', cell_type],
                    save=os.path.join(out_dir, 'umap_concat.png'),
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    resolution=resolution                    
            )
            concat_emb.write(f"{out_dir}/concat.h5ad")
        return concat_emb, rna_emb, atac_emb
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
