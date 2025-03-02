import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import json
import pandas as pd
import numpy as np
import scanpy as sc
import muon as mu


import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from scmamba2.preprocess import Preprocessor, scATACseqPreprocessor
from scmamba2.utils.process import lsi
from scmamba2.dataset.dataset import MultiomeDataset
from scmamba2.models import scMambaConfig, scMambaLMHeadModel
from scmamba2.loss import CLIPLoss, ContrastiveLoss
from scmamba2.trainer import Trainer
from scmamba2.utils.metrics import (
    biology_conservation, omics_mixing
)
from scmamba2 import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scMamba")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--checkpoint", type=str, 
        default=None
    )
    parser.add_argument("--Retraining", type=bool, default=True)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--gpu_ids", type=list, default=[0])

    # DataModule
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--data_dir", type=str, default="datasets/multiome/PBMC10k.h5mu"
    )
    parser.add_argument("--backed", action="store_true", default=False)
    parser.add_argument("--n_top_genes", type=int, default=1024)
    parser.add_argument("--n_top_peaks", type=int, default=2048)
    parser.add_argument("--LSI", type=bool, default=False)
    parser.add_argument("--PCA", type=bool, default=False)
    parser.add_argument("--mask", type=float, default=None)
    parser.add_argument("--binning", type=int, default=0)

    # Module
    parser.add_argument("--pool", type=str, default='last token')
    parser.add_argument("--config", type=str, default="config_files/scmamba2_config.json")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--requires_grad", action="store_true", default=True
    )
    parser.add_argument(
        "--multi_batches", action="store_true", default=False
    )
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cos_simi_scale", type=float, default=1)
    parser.add_argument("--epoch_nums", type=int, default=20)
    parser.add_argument("--results_dir", type=str, default='results/benckmark')
    
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True # for CNN
    torch.backends.cudnn.benchmark = False # for CNN
    torch.backends.cudnn.enabled = True # for CNN

    device = args.device
    gpu_ids = args.gpu_ids
    finally_epoch = 0
    
    mdata = mu.read(args.data_dir)
    # preprocess scRNA-seq dataset
    rna = mdata.mod['rna'].copy()
    preprocessor_rna = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=args.n_top_genes,  # 5. whether to subset the raw data to highly variable genes
        hvg_use_key=None,
        hvg_flavor="seurat_v3",
        binning=0,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor_rna(rna, batch_key=None if not args.multi_batches else 'batch')
    mdata.mod['rna'] = rna

    # preprocess scATAC-seq dataset
    atac = mdata.mod['atac'].copy()
    preprocessor_atac = scATACseqPreprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        binarize=True,  # 3. whether to binarize the raw data
        result_binarize_key="X_binarized", # the key in adata.layers to store the binarized data
        subset_hvg=args.n_top_peaks,  # 4. whether to subset the raw data to highly variable genes
        hvg_use_key=None,
        hvg_flavor="seurat_v3",
        binning=0,  # 5. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor_atac(atac, batch_key=None if not args.multi_batches else 'batch')
    mdata.mod['atac'] = atac
    mu.pp.intersect_obs(mdata)
    data_name = os.path.basename(args.data_dir).split('.')[0]
    if args.PCA and args.LSI:
        if os.path.exists(f'datasets/multiome/{data_name}_pca_2500.npy'):
            mdata['rna'].obsm['X_pca'] = \
                np.load(f'datasets/multiome/{data_name}_pca_2500.npy')
        else:
            sc.pp.pca(mdata['rna'], n_comps=2500)
        if os.path.exists(f'datasets/multiome/{data_name}_lsi_2500.npy'):
            mdata['atac'].obsm['X_lsi'] = \
                np.load(f'datasets/multiome/{data_name}_lsi_2500.npy')
        else:
            lsi(mdata['atac'], n_components=2500, n_iter=15)

        d_rna_feature = mdata.mod['rna'].obsm['X_pca'].shape[1]
        d_atac_feature = mdata.mod['atac'].obsm['X_lsi'].shape[1]
    else:
        d_rna_feature = mdata.mod['rna'].X.shape[1]
        d_atac_feature = mdata.mod['atac'].X.shape[1]

    # Prepare dataset
    
    train_dataset = MultiomeDataset(
        mdata, 
        "X_pca" if not args.binning else 'X_binned', 
        "X_binarized" if not args.LSI else 'X_lsi'
    )

    # load configure file
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
        device=device
    ).to(device)

    if torch.cuda.device_count() > 1: 
        print("Let's use", len(gpu_ids), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # build loss function and optimizer
    criterion = ContrastiveLoss(
        temperature=args.temperature, cos_simi_scale=args.cos_simi_scale
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # set up output directories and logging
    data_name = os.path.basename(args.data_dir).split('.')[0]
    out_dir = os.path.join(args.results_dir, data_name)
    out_dir = f"{out_dir}batchsize{args.batch_size}projection_dim{config_decoder1.d_embedding}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    writer = SummaryWriter(f'{out_dir}/runs/exp')
    
    if args.checkpoint is None:     # training model
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epochs=args.epoch_nums,
            checkpoint_dir=out_dir,
            writer=writer,
            device=args.device,
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        logger.info("Training the model ...")
        finally_epoch = trainer.fit(
            train_loader=train_dataloader,
        )

    elif args.Retraining:       # load checkpoint and re-training model
        # load the pre-training model
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epochs=args.epoch_nums,
            checkpoint_dir=out_dir,
            writer=writer,
            device=args.device,
            init_epoch=epoch
        )
        # train the model
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        logger.info("Training the model ...")
        finally_epoch = trainer.fit(
            train_loader=train_dataloader,
        )

    else:       # load checkpoint for inference
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        finally_epoch = checkpoint['epoch']

    if not args.fast_dev_run:
        model.eval()        
        # Test the model
        test_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        with torch.no_grad():
            concate_emb, rna_emb, atac_emb = model.get_representation(
                dataloader=test_loader, 
                cell_type='cell_type', 
                out_dir=out_dir, 
                device=device,
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

            metrics['epcohs'] = finally_epoch
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