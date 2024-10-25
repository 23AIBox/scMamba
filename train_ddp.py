import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import json
import pandas as pd
import numpy as np
import muon as mu
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from tensorboardX import SummaryWriter

from scmamba2.preprocess import Preprocessor, scATACseqPreprocessor
from scmamba2.dataset.dataset import MultiomeModule
from scmamba2.models import MambaLMHeadModel, MambaConfig, scMambaLMHeadModel
from scmamba2.loss import CLIPLoss
from scmamba2.trainer import Trainer
from scmamba2.utils.metrics import (
    biology_conservation, omics_mixing, mean_F1_silhouette
)
from scmamba2 import logger

def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up the DDP environment."""
    dist.destroy_process_group()

def main(rank, world_size, args):
    dist.init_process_group("nccl")
    # rank = dist.get_rank()
    # print(f"Start running basic DDP example on rank {rank}.")

    setup_ddp(rank, world_size)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # Load data
    mdata = mu.read(args.data_dir)
    rna = mdata.mod['rna'].copy()
    preprocessor_rna = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=args.n_top_genes,
        hvg_use_key=None,
        hvg_flavor="seurat_v3",
        binning=0,
        result_binned_key="X_binned",
    )
    preprocessor_rna(rna, batch_key=None)
    mdata.mod['rna'] = rna

    atac = mdata.mod['atac'].copy()
    preprocessor_atac = scATACseqPreprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        binarize=True,
        result_binarize_key="X_binarized",
        subset_hvg=args.n_top_peaks,
        hvg_use_key=None,
        hvg_flavor="seurat_v3",
        binning=0,
        result_binned_key="X_binned",
    )
    preprocessor_atac(atac, batch_key=None)
    mdata.mod['atac'] = atac
    mu.pp.intersect_obs(mdata)
    
    d_rna_feature = mdata.mod['rna'].X.shape[1]
    d_atac_feature = mdata.mod['atac'].X.shape[1]

    with open(args.config, 'r') as file:
        config = json.load(file)
    config = MambaConfig(**config)

    # Create model
    model = scMambaLMHeadModel(
        config=config,
        d_feature_omics1=d_rna_feature,
        d_feature_omics2=d_atac_feature,
        patch_size=512,
        device=rank
    ).to(rank)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = CLIPLoss(
        requires_grad=args.requires_grad, logit_scale=args.logit_scale
    )
    optimizer = optim.AdamW(
        ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Set up output directories and logging
    # data_name = os.path.basename(args.data_dir).split('.')[0]
    # out_dir = os.path.join(args.results_dir, data_name)
    # out_dir = f"{out_dir}batchsize{args.batch_size}projection_dim{args.projection_dim}"
    # os.makedirs(out_dir, exist_ok=True)
    # if not args.normalize:
    #     out_dir = f"{out_dir}_no_norm"
    # os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    
    # if rank == 0:
    #     writer = SummaryWriter(f'{out_dir}/runs/exp')

    # Prepare data loaders with DistributedSampler
    dm = MultiomeModule(mdata, "X_log1p", "X_binarized", num_workers=args.num_workers)
    dm.setup(stage="fit")
    
    train_sampler = DistributedSampler(dm.train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(dm.val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(dm.train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dm.val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    
    # Initialize Trainer and start training
    if rank == 0:
        logger.info("Training the model ...")
    

    ddp_model.train()
    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    for i, (rna, atac) in enumerate(loop):
        rna, atac = rna.float().to(args.device), atac.float().to(args.device)
        optimizer.zero_grad()

        CausalLMOutput = ddp_model(rna, atac, num_last_tokens=1)
        rna_embeds, atac_embeds = CausalLMOutput.logits_omics_1, CausalLMOutput.logits_omics_2

        loss, similarity = criterion(rna_embeds, atac_embeds)
        loss.backward()
        optimizer.step()

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scMamba with DDP")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--retrain", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--gpu_ids", type=list, default=[0, 1])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="datasets/multiome/PBMC10k.h5mu")
    parser.add_argument("--n_top_genes", type=int, default=10000)
    parser.add_argument("--n_top_peaks", type=int, default=20000)
    parser.add_argument("--config", type=str, default="mamba2attn_config.json")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--requires_grad", action="store_true", default=True)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--multi_batches", action="store_true", default=False)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--test_data", type=str, default="datasets/multiome/PBMC10k.h5mu")
    parser.add_argument("--logit_scale", type=float, default=1)
    parser.add_argument("--epoch_nums", type=int, default=80)
    parser.add_argument("--results_dir", type=str, default='scmamba2_results')
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs for DDP")
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
