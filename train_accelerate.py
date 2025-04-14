import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import json
import pandas as pd
import numpy as np
import muon as mu
import scanpy as sc
from tqdm import tqdm


from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration
from transformers import get_cosine_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

from scmamba2.preprocess import Preprocessor, scATACseqPreprocessor
from scmamba2.utils.process import lsi
from scmamba2.dataset.dataset import MultiomeDataset
from scmamba2.models.scmamba import scMambaLMHeadModel
from scmamba2.models.config_scmamba import scMambaConfig
from scmamba2.loss import CLIPLoss, ContrastiveLoss
from scmamba2.trainer import Trainer
from scmamba2.utils.metrics import (
    biology_conservation, omics_mixing, mean_F1_silhouette
)
from scmamba2.utils.plot import plot_paired_umap
from scmamba2 import logger


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # Load data
    mdata = mu.read(args.data_dir)
    if args.cell_numbers:
        selected_cells = np.random.choice(
            mdata.obs.index, size=args.cell_numbers, replace=False
        )
        mdata = mdata[selected_cells, :].copy()
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
        binning=args.binning,
        result_binned_key="X_binned",
    )
    preprocessor_rna(rna, batch_key=None)
    mdata.mod['rna'] = rna

    atac = mdata.mod['atac'].copy()
    preprocessor_atac = scATACseqPreprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        tfidf=False,
        result_tfidf_key="X_tfidf",
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
    
    # whether use PCA and LSI to process X matrix
    data_name = os.path.basename(args.data_dir).split('.')[0]
    data_path = os.path.dirname(args.data_dir)
    if args.PCA and args.LSI:
        if os.path.exists(f'{data_path}/{data_name}_pca_{args.PCA}.npy'):
            mdata['rna'].obsm['X_pca'] = \
                np.load(f'{data_path}/{data_name}_pca_{args.PCA}.npy')
        else:
            sc.pp.pca(mdata['rna'], n_comps=args.PCA)
            np.save(f'{data_path}/{data_name}_pca_{args.PCA}.npy', mdata.mod['rna'].obsm['X_pca'])
        if os.path.exists(f'{data_path}/{data_name}_lsi_{args.LSI}.npy'):
            mdata['atac'].obsm['X_lsi'] = \
                np.load(f'{data_path}/{data_name}_lsi_{args.LSI}.npy')
        else:
            lsi(mdata['atac'], n_components=args.LSI, n_iter=15)
            np.save(f'{data_path}/{data_name}_lsi_{args.LSI}.npy', mdata.mod['atac'].obsm['X_lsi'])

        d_rna_feature = mdata.mod['rna'].obsm['X_pca'].shape[1]
        d_atac_feature = mdata.mod['atac'].obsm['X_lsi'].shape[1]
    else:
        d_rna_feature = mdata.mod['rna'].X.shape[1]
        d_atac_feature = mdata.mod['atac'].X.shape[1]
    # d_rna_feature = mdata.mod['rna'].X.shape[1]
    # d_atac_feature = mdata.mod['atac'].X.shape[1]
    
    # Prepare data loaders
    train_dataset = MultiomeDataset(
        mdata, 
        "X_log1p" if not args.PCA else 'X_pca', 
        "X_binarized" if not args.LSI else 'X_lsi'
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    with open(args.config, 'r') as file:
        config = json.load(file)
    config_decoder1 = scMambaConfig(**config['decoder1'])
    config_decoder2 = scMambaConfig(**config['decoder2'])

    # Create model
    model = scMambaLMHeadModel(
        config_omics1=config_decoder1,
        config_omics2=config_decoder2,
        d_feature_omics1=d_rna_feature,
        d_feature_omics2=d_atac_feature,
        pool=args.pool,
        normalize=args.normalize
    )
    # Loss and optimizer
    criterion = CLIPLoss(
        requires_grad=args.requires_grad, logit_scale=args.logit_scale
    )
    criterion = ContrastiveLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # WarmupCosineLR
    num_training_steps = len(train_dataloader) * args.epoch_nums
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% steps for warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Set up output directories and logging
    data_name = os.path.basename(args.data_dir).split('.')[0]
    out_dir = os.path.join(args.results_dir, data_name)
    out_dir = f"{out_dir}batchsize{args.batch_size}projection_dim{config_decoder1.d_embedding}"
    checkpoints_path = os.path.join(out_dir, 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    writer = SummaryWriter(f'{out_dir}/runs/exp')

    deepspeed_plu = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
    project_config = ProjectConfiguration(project_dir=out_dir, automatic_checkpoint_naming=True)
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plu, project_config=project_config)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    
    logger.info("Training the model ...")
    for epoch in range(args.epoch_nums):
        loop = tqdm(train_dataloader, total=len(train_dataloader), leave=False)
        runing_loss = 0.0
        avg_loss = 0.0
        for step, (rna, atac) in enumerate(loop):            
            optimizer.zero_grad()

            rna, atac = rna.float(), atac.float()
            rna_embeds, atac_embeds = model(rna, atac)

            loss, _ = criterion(rna_embeds, atac_embeds)
            runing_loss += loss.item()
            avg_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            if (step + 1) % 30 == 0:
                writer.add_scalar("train loss (30 avg)", avg_loss / 30, epoch * len(train_dataloader) + step)
                avg_loss = 0.0
            
            loop.set_description(f'Epoch [{epoch}/{args.epoch_nums}]')
            loop.set_postfix(loss=loss.item())
        writer.add_scalar("train loss", runing_loss / len(train_dataloader), epoch)
        
        if (epoch + 1) % 10 == 0:
            accelerator.save_state(output_dir=checkpoints_path)
        writer.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    model = accelerator.unwrap_model(model)
    logger.info("saving the model's checkpoint ...")
    if accelerator.is_main_process:
        torch.save({
            "model_state_dict": model.state_dict(),
        }, f"{out_dir}/checkpoints/scMamba.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scMamba")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--retrain", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="datasets/multiome/fetal.h5mu")
    parser.add_argument("--n_top_genes", type=int, default=10240)
    parser.add_argument("--n_top_peaks", type=int, default=20480)
    parser.add_argument("--PCA", type=int, default=0)
    parser.add_argument("--LSI", type=int, default=0)
    parser.add_argument("--cell_numbers", type=int, default=0)
    parser.add_argument("--binning", type=int, default=0)
    parser.add_argument("--pool", type=str, default='last token')
    parser.add_argument("--config", type=str, default="config_files/scmamba2_config.json")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--requires_grad", action="store_true", default=True)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--logit_scale", type=float, default=1)
    parser.add_argument("--epoch_nums", type=int, default=100)
    parser.add_argument("--results_dir", type=str, default='results/accelerate_results')
    
    args = parser.parse_args()
    main(args)
