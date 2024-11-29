r"""
Performance evaluation metrics
"""
import os
import sys
sys.path.append("..")
from typing import Tuple
from threadpoolctl import threadpool_limits
import torch
import scipy
import cupy as cp
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

from scmamba2.utils.evaluation import (
    avg_silhouette_width_cell_type, mean_average_precision, neighbor_consistency,
    avg_silhouette_width_omics, seurat_alignment_score, graph_connectivity,
    maxARI, NMI, F1_silhouette
)


def matching_metrics(similarity=None, x=None, y=None, **kwargs):
    if similarity is None:
        if x.shape != y.shape:
            raise ValueError("Shapes do not match!")
        similarity = 1 - scipy.spatial.distance_matrix(x, y, **kwargs)
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.from_numpy(similarity)

    with torch.no_grad():
        # similarity = output.logits_per_atac
        batch_size = similarity.shape[0]
        acc_x = (
            torch.sum(
                torch.argmax(similarity, dim=1)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        acc_y = (
            torch.sum(
                torch.argmax(similarity, dim=0)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        foscttm_x = (
            (similarity > torch.diag(similarity)).float().mean(axis=1).mean().item()
        )
        foscttm_y = (
            (similarity > torch.diag(similarity)).float().mean(axis=0).mean().item()
        )
        # matchscore_x = similarity.softmax(dim=1).diag().mean().item()
        # matchscore_y = similarity.softmax(dim=0).diag().mean().item()
        X = similarity
        mx = torch.max(X, dim=1, keepdim=True).values
        hard_X = (mx == X).float()
        logits_row_sums = hard_X.clip(min=0).sum(dim=1)
        matchscore = hard_X.clip(min=0).diagonal().div(logits_row_sums).mean().item()

        acc = (acc_x + acc_y) / 2
        foscttm = (foscttm_x + foscttm_y) / 2
        # matchscore = (matchscore_x + matchscore_y)/2
        return acc, matchscore, foscttm

def compute_matching_score(embeddings_1, embeddings_2):
    """
    calculate matching Score
    
    parameters:
    embeddings_1 (np.ndarray): modality 1's embeddings, shape: (N, d), N is the number of samples, d is dimension
    embeddings_2 (np.ndarray): modality 2's embeddings, shape: (N, d), N is the number of samples, d is dimension
    
    return:
    matching_score (float): Matching Score
    """
    N = embeddings_1.shape[0]
    
    # calculate the nearest neighbor of modality 1 in the modality 2
    knn_idx_1 = np.argmin(np.linalg.norm(embeddings_1[:, None] - embeddings_2, axis=-1), axis=-1)
    # calculate the nearest neighbor of modality 2 in the modality 1
    knn_idx_2 = np.argmin(np.linalg.norm(embeddings_2[:, None] - embeddings_1, axis=-1), axis=-1)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            knn_ij = {knn_idx_1[i]}
            knn_jj = {j}
            knn_ji = {knn_idx_2[j]}
            knn_ii = {i}
            M[i, j] = len((knn_ij & knn_jj) | (knn_ji & knn_ii)) / len((knn_ij | knn_jj) | (knn_ji | knn_ii))

    # normalize matching matrix
    M_norm = M / (M.sum(axis=1, keepdims=True) + M.sum(axis=0, keepdims=True) - M)

    # calculate matching Score
    matching_score = (1 / N) * np.sum(np.diagonal(M_norm))
    return matching_score

def compute_foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = ((d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)).mean()
    foscttm_y = ((d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)).mean()
    return (foscttm_x + foscttm_y) / 2

def calculate_metrics(
        adata: sc.AnnData, 
        cell_type,
        cluster_labels,
        num_threads=16
):
    label_encoder = LabelEncoder()
    encoded_original_labels = label_encoder.fit_transform(cell_type)
    
    # biology conservation
    with threadpool_limits(limits=num_threads):
        ari_score = metrics.adjusted_rand_score(encoded_original_labels, cluster_labels)
        nmi_score = metrics.normalized_mutual_info_score(encoded_original_labels, cluster_labels)
        MAP = mean_average_precision(adata.X, cell_type)
        ASW_celltype = avg_silhouette_width_cell_type(adata.X, cell_type)
        
        # omics mixing
        BEM_score = batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['modality'])
        SAS = seurat_alignment_score(adata.obsm['X_umap'], cell_type)
        GC = graph_connectivity(adata.X, cell_type)
        ASW_omics = avg_silhouette_width_omics(adata.X, adata.obs['modality'].values, cell_type)

    return {
        "ARI": float(ari_score),
        "NMI": float(nmi_score),
        "Mean average precision": float(MAP),
        "ASW_celltype": float(ASW_celltype),
        
        "omics entropy mixing score": float(BEM_score),
        "Seurat alignment score (omics)": float(SAS),
        "Graph connectivity": float(GC),
        "ASW_omics": float(ASW_omics),
    }

def omics_mixing(
        adata: sc.AnnData, 
        cell_type,
        modality,
        num_threads=16
):
    
    with threadpool_limits(limits=num_threads):
        BEM_score = batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['modality'])
        SAS = seurat_alignment_score(adata.obsm['X_umap'], modality)
        GC = graph_connectivity(adata.obsm['X_umap'], cell_type)
        ASW_omics = avg_silhouette_width_omics(adata.obsm['X_umap'], modality, cell_type)

    return {
        "omics entropy mixing score": float(BEM_score),
        "Seurat alignment score (omics)": float(SAS),
        "Graph connectivity": float(GC),
        "ASW_omics": float(ASW_omics)
    }

def biology_conservation(
        adata: sc.AnnData, 
        cell_type,
        num_threads=16
):  
    # biology conservation
    with threadpool_limits(limits=num_threads):
        max_ari, max_res = maxARI(adata, cell_type="cell_type")
        sc.tl.leiden(adata, resolution=max_res, flavor="igraph", n_iterations=2)
        nmi_score = NMI(adata.obs['cell_type'].values, adata.obs['leiden'].values)
        # ari_score = metrics.adjusted_rand_score(encoded_original_labels, cluster_labels)
        # nmi_score = metrics.normalized_mutual_info_score(encoded_original_labels, cluster_labels)
        MAP = mean_average_precision(adata.X, cell_type)
        ASW_celltype = avg_silhouette_width_cell_type(adata.obsm['X_umap'], cell_type)

    return {
        "ARI": float(max_ari),
        "NMI": float(nmi_score),
        "Mean average precision": float(MAP),
        "ASW_celltype": float(ASW_celltype),
    }, max_res

def remove_batch_effects(
        adata, 
        cell_type,
        batch,
        num_threads=16
):    
    with threadpool_limits(limits=num_threads):
        BEM_score = batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['batch'])
        SAS = seurat_alignment_score(adata.X, batch)
        GC = graph_connectivity(adata.X, cell_type)
        ASW_batch = avg_silhouette_width_omics(adata.X, batch, cell_type)

    return {
        "batch entropy mixing score": float(BEM_score),
        "Seurat alignment score (batch)": float(SAS),
        "Graph connectivity": float(GC),
        "ASW_batch": float(ASW_batch)
    }    

def mean_F1_silhouette(embeds, cell_type, omics, device_id=0, chunk_size=1000):
    F1_sil_cuda = F1_silhouette(
        embeds, cell_type, omics, device_id=device_id, chunk_size=chunk_size
        )
    F1_sil_cuda_cpu = cp.asnumpy(F1_sil_cuda)
    return F1_sil_cuda_cpu.mean()

def batch_entropy_mixing_score(
        data, 
        batches, 
        n_neighbors=100, 
        n_pools=100, 
        n_samples_per_pool=100
    ):
    """
    Calculate batch entropy mixing score
    
    Algorithm
    -----
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.
    
    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.
        
    Returns
    -------
    Batch entropy mixing score
    """
    n_neighbors = int(len(data) * 0.01)
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log2(adapt_p[i]+10**-8)
        return entropy
    
    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    score = 0
    for t in range(n_pools):
        # Randomly choose `n_samples_per_pool` cells from all batches.
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        curr_score = 0
        for i in range(n_samples_per_pool):
            indice = indices[i]
            # Calculate the `n_neighbors` nearest neighbors for each randomly chosen cell.
            curr_n_neighbors = kmatrix[indice].nonzero()[1]         
            curr_score += entropy(batches.iloc[curr_n_neighbors])
        curr_score = curr_score / float(n_samples_per_pool)
        score += curr_score
    score = score / float(n_pools)
    return score / float(np.log2(N_batches))
