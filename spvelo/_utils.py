from pathlib import Path
from typing import Optional, Union, Callable, Any
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.spatial import Delaunay
from scanpy.neighbors import _get_indices_distances_from_dense_matrix
from scipy import sparse
import scipy.stats
from scvelo.preprocessing.neighbors import _get_rep, _set_pca, get_duplicate_cells, _set_neighbors_data
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import os, random
import pytorch_lightning
import squidpy as sq
import torch

import sklearn.neighbors

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj

def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    pytorch_lightning.seed_everything(seed)
    
def add_velovi_outputs_to_adata(adata, vae):
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

    t = latent_time
    scaling = 20 / t.max(0)
    scaling = scaling.to_numpy()

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0
    
def concat_pp(adatas_dict, concat_order, filter_nc=True):
    adatas = {}
    for sample in concat_order:
        adata = adatas_dict[sample]
        adata.obs['batch'] = [str(sample)]*adata.shape[0]
        adata.obs_names = [c+'_'+str(sample) for c in adata.obs_names.tolist()]
        adata.var_names_make_unique()
        if filter_nc == True:
            adata = adata[np.where(adata.obs.cluster_annotations!='nc')]
        adata.layers['spliced_counts'] = adata.layers['spliced'].copy()
        adata.layers['unspliced_counts'] = adata.layers['unspliced'].copy()
        scv.pp.filter_genes(adata, min_shared_counts=20)
        scv.pp.normalize_per_cell(adata)
        scv.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=30)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        sc.tl.umap(adata)
        sq.gr.spatial_neighbors(adata, spatial_key='spatial', n_neighs=30)
        adatas[sample] = adata.copy()
    adata = anndata.concat([adatas[sample] for sample in concat_order])
    return adata

def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
    r2_thresh=0.2,
) -> AnnData:
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data. From veloVI.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit
    r2_thresh
        Threshold to r2 for gene filtering

    Returns
    -------
    Preprocessed adata.
    """
    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > r2_thresh, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    return adata

def extract_edge_index(
    adata: AnnData,
    batch_key: Optional[str] = 'batch',
    spatial_key: Optional[str] = 'spatial',
    method: str = 'knn',
    n_neighbors: int = 15
    ):
    """
    Define edge_index for GAT encoder. From SIMVI.

    Args:
    ----
        adata: AnnData object.
        batch_key: Key in `adata.obs` for batch information. If batch_key is none,
        assume the adata is from the same batch. Otherwise, we create edge_index
        based on each batch and concatenate them.
        spatial_key: Key in `adata.obsm` for spatial location.
        method: method for establishing the graph proximity relationship between
        cells. Two available methods are: knn and Delouney. Knn is used as default
        due to its flexible neighbor number selection.
        n_neighbors: The number of n_neighbors of knn graph. Not used if the graph
        is based on Delouney triangularization.

    Returns
    -------
        edge_index: torch.Tensor.
    """
    if batch_key is not None:
        j = 0
        for i in adata.obs[batch_key].unique():
            adata_tmp = adata[adata.obs[batch_key]==i].copy()
            if method == 'knn':
                A = kneighbors_graph(adata_tmp.obsm[spatial_key],n_neighbors = n_neighbors)
                # A =  faiss_neig.knn_graph(adata_tmp.obsm[spatial_key],n_neighbors = n_neighbors)
                edge_index_tmp, edge_weight = from_scipy_sparse_matrix(A)
                label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                edge_index_tmp = label[edge_index_tmp]
                if j == 0:
                    edge_index = edge_index_tmp
                    j = 1
                else:
                    edge_index = torch.cat((edge_index,edge_index_tmp),1)

            else:
                tri = Delaunay(adata_tmp.obsm[spatial_key])
                triangles = tri.simplices
                edges = set()
                for triangle in triangles:
                    for i in range(3):
                        edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                        edges.add(edge)
                edge_index_tmp = torch.tensor(list(edges)).t().contiguous()
                label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                edge_index_tmp = label[edge_index_tmp]
                if j == 0:
                    edge_index = edge_index_tmp
                    j = 1
                else:
                    edge_index = torch.cat((edge_index,edge_index_tmp),1)
    else:
        if method == 'knn':
            # print(adata)
            # print(adata.obsm)
            A = kneighbors_graph(adata.obsm[spatial_key],n_neighbors = n_neighbors)
            # A =  faiss_neig.knn_graph(adata.obsm[spatial_key],n_neighbors = n_neighbors)
            edge_index, edge_weight = from_scipy_sparse_matrix(A)
        else:
            tri = Delaunay(adata.obsm[spatial_key])
            triangles = tri.simplices
            edges = set()
            for triangle in triangles:
                for i in range(3):
                    edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                    edges.add(edge)
            edge_index = torch.tensor(list(edges)).t().contiguous()

    return edge_index


def new_BatchKNN(adata, batch_key, nn_key, k=8, use_rep='pca'):
    kNNGraphIndex = np.zeros(shape=(adata.shape[0], k))
    kNNGraphWeight = np.zeros(shape=(adata.shape[0], k))

    for val in np.unique(adata.obs[batch_key]):
        val_ind = np.where(adata.obs[batch_key] == val)[0]
        
        if nn_key == None:
            batch_knn = sklearn.neighbors.kneighbors_graph(
                adata[val_ind].X, n_neighbors=k, mode="connectivity", n_jobs=-1
            ).tocoo()
            batch_dist = sklearn.neighbors.kneighbors_graph(
                adata[val_ind].X, n_neighbors=k, mode="distance", n_jobs=-1
            ).tocoo()       
        
        elif nn_key in adata.obsm.keys():
            batch_knn = sklearn.neighbors.kneighbors_graph(
                adata[val_ind].obsm[nn_key], n_neighbors=k, mode="connectivity", n_jobs=-1
            ).tocoo()
            batch_dist = sklearn.neighbors.kneighbors_graph(
                adata[val_ind].obsm[nn_key], n_neighbors=k, mode="distance", n_jobs=-1
            ).tocoo()
        
        elif nn_key in adata.layers.keys():
            batch_knn = sklearn.neighbors.kneighbors_graph(
                adata[val_ind].layers[nn_key], n_neighbors=k, mode="connectivity", n_jobs=-1
            ).tocoo()
            batch_dist = sklearn.neighbors.kneighbors_graph(
                adata[val_ind].layers[nn_key], n_neighbors=k, mode="distance", n_jobs=-1
            ).tocoo()
        
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn.col), [adata[val_ind].shape[0], k]
        )
        kNNGraphWeight[val_ind] = np.array(batch_dist.data).reshape(adata[val_ind].shape[0], k)
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]

    return kNNGraphIndex.astype("int"), kNNGraphWeight

def aggr_niche(adata, idx, weight):
    new_count = np.zeros(shape=adata.shape)
    norm_weight = weight/np.sum(weight, axis=1, keepdims=True)
    for i in range(adata.shape[0]):
        new_count[i] = np.array(adata.X[i])[0] + norm_weight[i].reshape(1,-1) @ np.array(adata[idx[i]].X)
    adata.layers['aggr'] = new_count
    return adata
