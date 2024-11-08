import os
import torch
import muon as mu
import scanpy as sc
import episcanpy as epi
import numpy as np
import scipy
from anndata import AnnData
from typing import List, Tuple, Dict, Optional, Union
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from sklearn.preprocessing import maxabs_scale, LabelEncoder



class MultiomeDataset(Dataset):
    def __init__(self, mdata: mu.MuData, use_layer1: str, use_layer2: str) -> None:
        super().__init__()
        self.mdata = mdata
        self.use_layer1 = use_layer1
        self.use_layer2 = use_layer2
    
    def __len__(self):
        return self.mdata['rna'].X.shape[0]
    
    def __getitem__(self, index):
        if self.use_layer1 == "X_pca":
            x = self.mdata["rna"].obsm["X_pca"][index]
        elif self.use_layer1 != "X":
            x = self.mdata["rna"].layers[self.use_layer1][index].toarray().squeeze()
        else:
            x = self.mdata["rna"].X[index].toarray().squeeze()
        
        if self.use_layer2 == "X_lsi":
            y = self.mdata["atac"].obsm["X_lsi"][index]
        elif self.use_layer2 != "X":
            y = self.mdata["atac"].layers[self.use_layer2][index].toarray().squeeze()
        else:
            y = self.mdata["atac"].X[index].toarray().squeeze()
            
        return x, y
    
class MultiomeModule():
    def __init__(self,
            mdata: mu.MuData,
            use_layer1: str, 
            use_layer2: str,
            batch_size: int = 256,
            num_workers: int = 4,
            pin_memory: bool = False,
        ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = MultiomeDataset(mdata, use_layer1, use_layer2)

        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = self.dataset

        if stage == "predict":
            self.predict_dataset = self.dataset

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def predict_dataloader(self, dataset=None, shuffle=False):
        loader = DataLoader(
            dataset or self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader