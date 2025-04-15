import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import json
import pandas as pd 
import numpy as np
import muon as mu

import torch
from torch.utils.data import DataLoader

from scmamba2.preprocess import Preprocessor
from scmamba2.dataset.dataset import MultiomeDataset
from scmamba2.models import scMambaConfig
from scmamba2.models.scmamba import scMambaLMHeadModel
from scmamba2.utils.metrics import (
    biology_conservation, omics_mixing
)
from scmamba2 import logger


def main(args):
    torch.cuda.set_device(args.device)
    
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
    ).to(args.device)

    checkpoint = torch.load(args.checkpoints)
    model.load_state_dict(checkpoint['model_state_dict'])

    data_name = os.path.basename(args.data_dir).split('.')[0]
    out_dir = os.path.join(args.results_dir, data_name)
    out_dir = f"{out_dir}batchsize{args.batch_size}emb_dim{config_decoder1.d_embedding}"
    os.makedirs(out_dir, exist_ok=True)

    dataset = MultiomeDataset(
        mdata, 
        "X_binned" if args.binning else 'X_log1p', 
        "X_binned" if args.binning else 'X', 
        omics1='rna', 
        omics2='adt'
    )
    # Test the model
    test_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    
    model.eval()    
    with torch.no_grad():
        concate_emb, rna_emb, atac_emb = model.get_representation(
            dataloader=test_loader, 
            cell_type='cell_type', 
            out_dir=out_dir, 
            device=args.device,
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
        logger.info("Calculating omics alignment metrics...")
        omics_mixing_metrics = omics_mixing(
            concate_emb, concate_emb.obs['cell_type'].values, concate_emb.obs['modality'].values
        )
        metrics_classified['biology conservation'] = biology_conservation_metrics
        metrics_classified['omics alignment'] = omics_mixing_metrics
        
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
        "--checkpoints", type=str, 
        default=None
    )
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--gpu_ids", type=list, default=[0, 1])
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="batch size to be processed by one GPU in one step")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--data_dir", type=str, default="datasets/multiome/citeseq_BMMC_S1.h5mu")
    parser.add_argument("--batch_key", type=str, default=None)
    parser.add_argument("--n_top_genes", type=int, default=0)
    parser.add_argument("--binning", type=int, default=0)
    parser.add_argument("--config", type=str, default="config_files/scmamba2attn_config_rna_adt.json")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--requires_grad", action="store_true", default=True)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--multi_batches", action="store_true", default=False)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--cos_simi_scale", type=float, default=1)
    parser.add_argument("--epoch_nums", type=int, default=80)
    parser.add_argument("--results_dir", type=str, default='results/accelerate_results')
     
    args = parser.parse_args()

    main(args)   