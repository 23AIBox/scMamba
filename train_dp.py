import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import json
import time
import pandas as pd 
import numpy as np
import muon as mu
from tqdm import tqdm

import deepspeed
import deepspeed.comm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from tensorboardX import SummaryWriter

from scmamba2.preprocess import Preprocessor, scATACseqPreprocessor
from scmamba2.dataset.dataset import MultiomeModule, MultiomeDataset
from scmamba2.models import MambaLMHeadModel, MambaConfig, scMambaLMHeadModel
from scmamba2.loss import CLIPLoss
from scmamba2.trainer import Trainer
from scmamba2.utils.metrics import (
    biology_conservation, omics_mixing, mean_F1_silhouette
)
from scmamba2 import logger


def get_deepspeed_config():
    return {
        # "train_batch_size": 255,
        "train_micro_batch_size_per_gpu": 128,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1.5e-4,
                "betas": [
                    0.9,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 0.05,
            }
        },
        "fp16": {
            "enabled": False,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },        
        "zero_optimization": {
            "stage": 3,
        },
        "comms_logger": {
            "enabled": True,
            "verbose": False,
            "prof_all": True,
            "debug": False
        },
    }

def main(args):
    # init distributed
    deepspeed.init_distributed()

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
    preprocessor_rna(rna, batch_key="batch")
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
    preprocessor_atac(atac, batch_key="batch")
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
        patch_size=256,
        multi_batches=args.multi_batches
    )
    # Loss and optimizer
    criterion = CLIPLoss(
        requires_grad=args.requires_grad, logit_scale=args.logit_scale
    )

    # Set up output directories and logging
    out_dir = os.path.join(args.results_dir, os.path.basename(args.data_dir).split('.')[0])
    out_dir = f"{out_dir}batchsize{args.batch_size}projection_dim{config.vocab_size}"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(f'{out_dir}/runs/exp')
    

    # Prepare data loaders with DistributedSampler
    train_dataset = MultiomeDataset(mdata, "X_log1p", "X_binarized")
    
    # init engine
    engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=get_deepspeed_config(),
    )

    if args.ckpt_id:
        _, client_sd = engine.load_checkpoint(args.load_dir, args.ckpt_id)
    
    else:
        # deepspeed.comm.all_reduce()
        logger.info("Training the model ...")
        engine.train()
        for epoch in range(args.epoch_nums):
            runing_loss = 0.0
            avg_loss = 0.0
            for step, (rna, atac) in enumerate(training_dataloader):
                pre = time.time()
                rna = rna.to(device=engine.local_rank, dtype=torch.float32)
                atac = atac.to(device=engine.local_rank, dtype=torch.float32)
                rna_embeds, atac_embeds = engine(rna, atac)

                loss, _ = criterion(rna_embeds, atac_embeds)
                runing_loss += loss.item()
                avg_loss += loss.item()
                # runs backpropagation
                engine.backward(loss)
                # weight update
                engine.step()
                post = time.time()
                if (step + 1) % 10 == 0:
                    avg_loss = avg_loss / 30
                    writer.add_scalar("train loss (10 avg)", avg_loss, epoch * len(training_dataloader) + step)
                    avg_loss = 0.0
                if (step + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch}/{args.epoch_nums}], step={step + 1}, loss={loss.item()}")
            
            if (epoch + 1) % 10 == 0:
                ckpt_id = f"scMamba{epoch + 1}"
                engine.save_checkpoint(ckpt_dir, ckpt_id)

            writer.add_scalar("train loss", runing_loss / len(training_dataloader), epoch)
            
            # deepspeed.comm.log_summary()
    if engine.local_rank == 0:
        data_name = os.path.basename(args.data_dir).split('.')[0]
        out_dir = os.path.join(args.results_dir, data_name)
        out_dir = f"{out_dir}batchsize{args.batch_size}projection_dim{config.vocab_size}"
        os.makedirs(out_dir, exist_ok=True)
        
        model = engine.module
        model.eval()
        
        # Test the model
        test_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        with torch.no_grad():
            concate_emb, rna_emb, atac_emb = model.get_representation(
                dataloader=test_loader, 
                cell_type='cell_type', 
                out_dir=out_dir, 
                device=engine.local_rank,
                n_neighbors=30, 
                metric='cosine', 
                min_dist=0.5, 
                resolution=0.3
            )
            metrics = {}
            metrics_classified = {}
            logger.info("Calculating biology conservation metrics...")
            biology_conservation_metrics, best_res= biology_conservation(
                concate_emb, concate_emb.obs['cell_type'].values
            )
            logger.info("Calculating omics mixing metrics...")
            omics_mixing_metrics = omics_mixing(
                concate_emb, concate_emb.obs['cell_type'].values, concate_emb.obs['modality'].values
            )
            metrics_classified['biology conservation'] = biology_conservation_metrics
            metrics_classified['omics mixing'] = omics_mixing_metrics
            
            metrics.update(biology_conservation_metrics)
            metrics.update(omics_mixing_metrics)

            metrics['epcohs'] = args.epoch_nums
            metrics['best resolutioin'] = best_res
            print(metrics)

            if not os.path.exists(f'{out_dir}/metrics.csv'):
                metrics_df = pd.DataFrame([metrics])
            else:
                metrics_df = pd.read_csv(
                    f'{out_dir}/metrics.csv' 
                )
                new_metrics_df = pd.DataFrame([metrics])
                metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)
            metrics_df.to_csv(f'{out_dir}/metrics.csv', index=False)        


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="scMamba")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--load_dir", type=str, 
        default=None
    )
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--retrain", type=bool, default=False)
    parser.add_argument("--results_dir", type=str, default='results/ds_results')

    # DataModule
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="batch size to be processed by one GPU in one step")
    parser.add_argument("--epoch_nums", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="datasets/multiome/multiome_BMMC.h5mu")
    parser.add_argument("--n_top_genes", type=int, default=20000)
    parser.add_argument("--n_top_peaks", type=int, default=40000)
    
    # Module
    parser.add_argument("--config", type=str, default="mamba2attn_config.json")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--requires_grad", action="store_true", default=True)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--multi_batches", action="store_true", default=True)
    parser.add_argument("--logit_scale", type=float, default=1)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')    
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
