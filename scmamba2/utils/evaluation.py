import warnings
import cupy as cp
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union
from threadpoolctl import threadpool_limits
RandomState = Optional[Union[np.random.RandomState, int]]


def ARI(
        cell_type, cluster_label
):
    """
    calculate the adjusted rand score
    """
    label_encoder = LabelEncoder()
    encoded_original_labels = label_encoder.fit_transform(cell_type)
    ARI_score = metrics.adjusted_rand_score(encoded_original_labels, cluster_label)
    
    return ARI_score

def maxARI(
        adata: sc.AnnData, cell_type="cell_type"  
):
    max_res = 0
    max_ARI = 0
    for res in range(1, 21):
        res = res/10.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sc.tl.leiden(adata, resolution=res, flavor="igraph", n_iterations=2)
            warnings.warn("This is a warning", UserWarning)
        ARI_score = ARI(adata.obs[cell_type].values, adata.obs["leiden"].values)
        if ARI_score > max_ARI:
            max_res = res
            max_ARI = ARI_score
    return max_ARI, max_res

def NMI(
        cell_type, cluster_label        
):
    """
    calculate the normalized mutual infomamtion score
    """
    label_encoder = LabelEncoder()
    encoded_original_labels = label_encoder.fit_transform(cell_type)
    NMI_score = metrics.normalized_mutual_info_score(encoded_original_labels, cluster_label)
    
    return NMI_score

def mean_average_precision(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Mean average precision

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    neighbor_frac
        Nearest neighbor fraction
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    map
        Mean average precision
    """
    k = max(round(y.shape[0] * neighbor_frac), 1)
    nn = NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), **kwargs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    return np.apply_along_axis(_average_precision, 1, match).mean().item()

def _average_precision(match: np.ndarray) -> float:
    if np.any(match):
        cummean = np.cumsum(match) / (np.arange(match.size) + 1)
        return cummean[match].mean().item()
    return 0.0

def avg_silhouette_width_cell_type(
        embedings: np.ndarray, cell_type: np.ndarray, **kwargs
) -> float:
    r"""
    Cell type average silhouette width

    Parameters
    ----------
    mebedings
        Coordinates
    cell_type
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_score`

    Returns
    -------
    asw
        Cell type average silhouette width

    """
    with threadpool_limits(limits=10):
        ASW_cell_type = (metrics.silhouette_score(embedings, cell_type, **kwargs).item() + 1) / 2
    return ASW_cell_type

def neighbor_consistency(
        x: np.ndarray, y: np.ndarray, batch: np.ndarray,
        neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Neighbor consistency score

    Parameters
    ----------
    x
        Cooordinates after integration
    y
        Coordinates before integration
    b
        Batch
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    nn_cons
        Neighbor consistency score
    """
    nn_cons_per_batch = []
    for b in np.unique(batch):
        mask = batch == b
        x_, y_ = x[mask], y[mask]
        k = max(round(x.shape[0] * neighbor_frac), 1)
        nnx = NearestNeighbors(
            n_neighbors=min(x_.shape[0], k + 1), **kwargs
        ).fit(x_).kneighbors_graph(x_)
        nny = NearestNeighbors(
            n_neighbors=min(y_.shape[0], k + 1), **kwargs
        ).fit(y_).kneighbors_graph(y_)
        nnx.setdiag(0)  
        nny.setdiag(0)  
        n_intersection = nnx.multiply(nny).sum(axis=1).A1
        n_union = (nnx + nny).astype(bool).sum(axis=1).A1
        nn_cons_per_batch.append((n_intersection / n_union).mean())
    return np.mean(nn_cons_per_batch).item()


def avg_silhouette_width_omics(
        x: np.ndarray, y: np.ndarray, ct: np.ndarray, **kwargs
) -> float:
    r"""
    Batch average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        omics labels
    ct
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_samples`

    Returns
    -------
    asw_batch
        Batch average silhouette width
    """
    s_per_ct = []
    for t in np.unique(ct):
        mask = ct == t
        try:
            s = metrics.silhouette_samples(x[mask], y[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
    return np.mean(s_per_ct).item()

def seurat_alignment_score(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01,
        n_repeats: int = 4, random_state: RandomState = None, **kwargs
) -> float:
    r"""
    Seurat alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = get_rs(random_state)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = NearestNeighbors(
            n_neighbors=k + 1, **kwargs
        ).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.mean(repeat_scores).item()

def get_rs(x: RandomState = None) -> np.random.RandomState:
    r"""
    Get random state object

    Parameters
    ----------
    x
        Object that can be converted to a random state object

    Returns
    -------
    rs
        Random state object
    """
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random


def graph_connectivity(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> float:
    r"""
    Graph connectivity

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`scanpy.pp.neighbors`

    Returns
    -------
    conn
        Graph connectivity
    """
    x = sc.AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X", **kwargs)
    conns = []
    for y_ in np.unique(y):
        x_ = x[y == y_]
        _, c = connected_components(
            x_.obsp['connectivities'],
            connection='strong'
        )
        counts = pd.value_counts(c)
        conns.append(counts.max() / counts.sum())
    return np.mean(conns).item()


def silhouette_samples_cuda(X, labels, device_id=0, chunk_size=1000):
    # 将数据转移到指定的GPU上，使用float32以减少显存
    with cp.cuda.Device(device_id):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        labels_gpu = cp.asarray(labels)

        # 获取数据点的数量
        n_samples = X.shape[0]
        
        # 初始化 a(i) 和 b(i)
        a = cp.zeros(n_samples, dtype=cp.float32)
        b = cp.zeros(n_samples, dtype=cp.float32)

        # 获取所有唯一的标签
        unique_labels = cp.unique(labels_gpu)

        # 分块计算距离矩阵
        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)

            # 计算块内的距离矩阵
            distance_matrix_chunk = cp.linalg.norm(X_gpu[i:end_i, None] - X_gpu, axis=2)

            # 逐个计算每个样本点的 a(i) 和 b(i)
            for j in range(end_i - i):
                current_label = labels_gpu[i + j]
                
                # 计算 a(i): 簇内距离平均值
                same_cluster_mask = (labels_gpu == current_label)
                same_cluster_distances = distance_matrix_chunk[j, same_cluster_mask]
                if cp.sum(same_cluster_mask) > 1:
                    a[i + j] = cp.sum(same_cluster_distances) / (cp.sum(same_cluster_mask) - 1)
                else:
                    a[i + j] = 0

                # 计算 b(i): 最近簇距离平均值
                b[i + j] = cp.inf
                for label in unique_labels:
                    if label != current_label:
                        other_cluster_mask = (labels_gpu == label)
                        other_cluster_distances = distance_matrix_chunk[j, other_cluster_mask]
                        if other_cluster_distances.size > 0:
                            b[i + j] = cp.minimum(b[i + j], cp.mean(other_cluster_distances))

        # 计算轮廓系数 s(i)
        s = (b - a) / cp.maximum(a, b)
        return s

def F1_silhouette(embeds, cell_type, omics, device_id=0, chunk_size=1000):
    # 将cell_type和omics转换为分类数据并转换为整数编码
    cell_type_encoded = pd.Categorical(cell_type).codes
    omics_encoded = pd.Categorical(omics).codes

    # 使用指定的GPU设备进行计算
    with cp.cuda.Device(device_id):
        # 确保数据在GPU上
        embeds_gpu = cp.asarray(embeds, dtype=cp.float32)
        cell_type_gpu = cp.asarray(cell_type_encoded)
        omics_gpu = cp.asarray(omics_encoded)
        
        # 计算每种标记的轮廓系数，使用分块
        s_ct = silhouette_samples_cuda(embeds_gpu, cell_type_gpu, device_id, chunk_size)
        s_omics = silhouette_samples_cuda(embeds_gpu, omics_gpu, device_id, chunk_size)
        
        # 归一化轮廓系数
        s_ct1 = (s_ct + 1) / 2
        s_omics1 = (s_omics + 1) / 2
        
        # 计算F1_silhouette
        F1_sil = 2 * ((1 - s_omics1) * s_ct1) / (1 - s_omics1 + s_ct1)
    return F1_sil