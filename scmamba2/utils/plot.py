import os
import numpy as np
import scanpy as sc
import pynndescent
import numba
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits


@numba.njit(fastmath=True)
def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result


pynn_dist_fns_fda = pynndescent.distances.fast_distance_alternatives
pynn_dist_fns_fda["cosine"]["correction"] = correct_alternative_cosine
pynn_dist_fns_fda["dot"]["correction"] = correct_alternative_cosine

def plot_umap(
    adata,
    color=None,
    save=None,
    n_neighbors=30,
    min_dist=0.5,
    metric="euclidean",
    resolution=1,
    use_rep="X",
    num_threads=16,
):
    """Plot a single UMAP plot"""
    sc.settings.set_figure_params(
        dpi=200, facecolor="white", figsize=(4, 4), frameon=True
    )
    with threadpool_limits(limits=num_threads):
        sc.pp.neighbors(adata, metric=metric, use_rep=use_rep, n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist, spread=1.0)
        sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    
    sc.pl.umap(adata, color=color, save=save, wspace=0.4, ncols=4)

def plot_paired_umap(
        adata,
        save=None,
        color: list = ['modality', 'cell_type'],
        n_neighbors=30,
        min_dist=0.5,
        resolution=1,
        metric="euclidean", 
        use_rep="X",
        num_threads=16,
):
    """Plot paired UMAP plots"""
    sc.settings.set_figure_params(
        dpi=300, facecolor="white", figsize=(4, 4), frameon=True
    )
    with threadpool_limits(limits=num_threads):
        sc.pp.neighbors(adata, metric=metric, use_rep=use_rep, n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist, spread=1.0)
        sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    
    ncols, nrows, figsize, wspace = 2, 1, 4, 0.5
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize + figsize * wspace * (ncols - 1), nrows * figsize),
    )
    plt.subplots_adjust(wspace=wspace)
    sc.pl.umap(adata, color=color[0], ax=axs[0], show=False, wspace=0.65, ncols=4)
    sc.pl.umap(adata, color=color[1], ax=axs[1], show=False, wspace=0.65, ncols=4)

    # axs[1, 0].axis('off')
    # sc.pl.umap(adata, color='leiden', ax=axs[1, 1], show=False, wspace=0.65, ncols=4)
    # legend = axs[1].get_legend()
    # legend.set_bbox_to_anchor((-1.53, -0.8))
    # legend.set_ncols(16)

    # for text in legend.get_texts():
    #     text.set_fontsize(8)

    fig.savefig(fname=save, bbox_inches='tight', dpi=500)
    print(f"saving figure to file {save}")
    return fig.figure