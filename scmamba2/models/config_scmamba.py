from dataclasses import dataclass, field


@dataclass
class scMambaConfig:

    d_model: int = 1024
    patch_size: int = 256
    d_intermediate: int = 1024
    n_layer: int = 12
    d_embedding: int = 64
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
