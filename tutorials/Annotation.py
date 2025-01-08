import warnings
# warnings.filterwarnings('ignore')

import sys
import os
import csv
# sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scanpy as sc
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

from scmamba2.models import ClsDecoder


def main(args):
    device = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    adata = sc.read_h5ad(args.data_dir)

    rna = adata[adata.obs['modality'] == 'rna'].copy()
    atac = adata[adata.obs['modality'] == 'atac'].copy()
    cell_emb = (rna.X + atac.X) / 2

    adata = sc.AnnData(
        X=cell_emb,
        obs=rna.obs,
    )
    adata = rna.copy()
    celltype = adata.obs['cell_type']
    train_emb = adata[adata.obs['batch'].isin(args.train_batches)].X
    train_cell_type = adata[adata.obs['batch'].isin(args.train_batches)].obs['cell_type']
    
    test_emb = adata[~adata.obs['batch'].isin(args.train_batches)].X
    test_cell_type = adata[~adata.obs['batch'].isin(args.train_batches)].obs['cell_type']
    
    # encoder the cell types
    label_encoder = LabelEncoder()
    celltype_encoded = label_encoder.fit_transform(celltype)
    train_cell_type_encoded = label_encoder.transform(train_cell_type)
    test_cell_type_encoded = label_encoder.transform(test_cell_type)
    # train_cell_type_encoded = label_encoder.fit_transform(train_cell_type)
    # test_cell_type_encoded = label_encoder.transform(test_cell_type)
    
    # transform to tensor
    X_train_tensor = torch.tensor(train_emb, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_cell_type_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(test_emb, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_cell_type_encoded, dtype=torch.long)

    # create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # init the model
    input_dim = train_emb.shape[1]
    output_dim = len(label_encoder.classes_)
    hidden_dim = 256
    model = ClsDecoder(d_model=input_dim, n_cls=output_dim, nlayers=3, hidden_dim=hidden_dim).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Set up output directories and logging
    out_dir = f"{args.results_path}batchsize{args.batch_size}"
    checkpoints_path = os.path.join(out_dir, 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    writer = SummaryWriter(f'{out_dir}/runs/exp')


    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        for step, (X_emb, cell_type) in enumerate(loop):
            X_emb, cell_type = X_emb.to(device), cell_type.to(device)
            optimizer.zero_grad()
            outputs = model(X_emb)
            loss = criterion(outputs, cell_type)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())

        writer.add_scalar("train loss", epoch_loss / len(train_loader), epoch)
        # print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            loop = tqdm(test_loader, total=len(test_loader), leave=False)
            for X_emb, cell_type in loop:
                X_emb, cell_type = X_emb.to(device), cell_type.to(device)
                outputs = model(X_emb)
                loss = criterion(outputs, cell_type)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += cell_type.size(0)
                correct += (predicted == cell_type).sum().item()

                loop.set_description(f'Validating Epoch [{epoch}/{args.epochs}]')
                loop.set_postfix(loss=loss.item(), accuracy=correct/cell_type.size(0))            
            
            accuracy = correct / total
            avg_loss = total_loss / len(test_loader)
        
        writer.add_scalar("val_loss", avg_loss, epoch)
        writer.add_scalar("val_accuracy", accuracy, epoch)
    
    writer.close()

    model.eval()
    all_predictions = []
    all_true = []
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(test_loader, total=len(test_loader), leave=False)
        for X_emb, cell_type in loop:
            X_emb, cell_type = X_emb.to(device), cell_type.to(device)
            outputs = model(X_emb)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true.extend(cell_type.cpu().numpy())
            total += cell_type.size(0)
            correct += (predicted == cell_type).sum().item()
    metrics = {}
    metrics['acc'] = correct / total
    metrics['f1_score'] = f1_score(all_true, all_predictions, average='weighted')
    metrics['precision'] = precision_score(all_true, all_predictions, average='weighted')
    metrics['recall'] = recall_score(all_true, all_predictions, average='weighted')
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

    
    # f1 = f1_score(all_true, all_predictions, average='weighted')
    # precision = precision_score(all_true, all_predictions, average='weighted')
    # recall = recall_score(all_true, all_predictions, average='weighted')

    decoded_predictions = label_encoder.inverse_transform(all_predictions)
    adata_subset = adata[~adata.obs['batch'].isin(args.train_batches)].copy()
    adata_subset.obs['cell_type_prediction'] = decoded_predictions
    adata.obs.loc[~adata.obs['batch'].isin(args.train_batches), 'cell_type_prediction'] = decoded_predictions

    # adata[~adata.obs['batch'].isin(args.train_batches)].obs['cell_type_prediction'] = decoded_predictions
    adata.write(f"{out_dir}/adata_prediction.h5ad")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scMamba")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="results/benckmark/human_brainbatchsize32projection_dim64/concat.h5ad"
    )
    parser.add_argument("--lr", type=float, default='1e-4')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    # s1d1 is the training batch for s1, s4d9 is training batch for s4
    parser.add_argument("--train_batches", type=list, default=['AD'])
    parser.add_argument("--results_path", type=str, default='results/annotation/human_brain')

    args = parser.parse_args()
    main(args)

