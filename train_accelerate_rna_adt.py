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

from scmamba2.preprocess import Preprocessor
from scmamba2.dataset.dataset import MultiomeDataset
from scmamba2.models import scMambaConfig
from scmamba2.models.scmamba import scMambaLMHeadModel
from scmamba2.loss import ContrastiveLoss
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
    preprocessor_rna(rna, batch_key=args.batch_key)
    mdata.mod['rna'] = rna

    protein = mdata.mod['adt'].copy()
    preprocessor_protein = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=0,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=False,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=False,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor=None,
        binning=args.binning,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor_protein(protein, batch_key=None)
    mdata.mod['adt'] = protein
    mu.pp.intersect_obs(mdata)
    
    d_rna_feature = mdata.mod['rna'].X.shape[1]
    d_adt_feature = mdata.mod['adt'].X.shape[1]
    
    # Prepare data loaders
    train_dataset = MultiomeDataset(
        mdata, 
        "X_binned" if args.binning else 'X_log1p', 
        "X_binned" if args.binning else 'X', 
        omics1='rna', 
        omics2='adt'
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
        d_feature_omics2=d_adt_feature,
        pool='first token'
    )
    # Loss and optimizer
    criterion = ContrastiveLoss(cos_simi_scale=args.cos_simi_scale)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # WarmupCosineLR
    num_training_steps = len(train_dataloader) * args.epoch_nums  # 假设训练 10 个 epoch
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% 的步数用于 warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Set up output directories and logging
    data_name = os.path.basename(args.data_dir).split('.')[0]
    out_dir = os.path.join(args.results_dir, data_name)
    out_dir = f"{out_dir}batchsize{args.batch_size}emb_dim{config_decoder1.d_embedding}"
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
        
        # if (epoch + 1) % 10 == 0:
        #     accelerator.save_state(output_dir=checkpoints_path)
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="datasets/multiome/cite_BMMC_S1.h5mu")
    parser.add_argument("--batch_key", type=str, default=None)
    parser.add_argument("--n_top_genes", type=int, default=0)
    parser.add_argument("--binning", type=int, default=0)
    parser.add_argument("--config", type=str, default="config_files/scmamba2attn_config_rna_adt.json")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--requires_grad", action="store_true", default=True)
    parser.add_argument("--multi_batches", action="store_true", default=False)
    parser.add_argument("--cos_simi_scale", type=float, default=1)
    parser.add_argument("--epoch_nums", type=int, default=100)
    parser.add_argument("--results_dir", type=str, default='results')
    
    args = parser.parse_args()
    main(args)
