import pandas as pd 
import scanpy as sc
import anndata
import numpy as np
import torch
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from .eval_util import *
from ._utils import *
import pytorch_lightning
import squidpy as sq
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse

def run_spVelo(adata, save_name, save_dir, method_name="spVelo", add_mnn=True, spatial_neighbors=15, mnn_neighbors=15, filter_uninfo=0.2, penalty='mmd', niche_edge=False, subsample=1.0, epoch=1000, batch_size=8192, batch_key='batch', spatial_key='spatial', ct_key='cluster_annotations'):
    '''Running spVelo.
    Parameters:
        adata: input multi-batch spatial transcriptomics dataset.
        save_name + method_name: saving name of the model.
        save_dir: saving directory of model and result adata with velocity and parameter estimate.
        add_mnn: A bool value used to determine whether to use batch information as input to GAT encoder.
        spatial_neighbors: number of spatial neighbors used in GAT encoder for modeling spatial location.
        mnn_neighbors: number of mnn used in GAT encoder for modeling batch information. Only used when setting add_mnn=True.
        filter_uninfo: If set as None, don't filter using R square. If set as a number, filter genes with R square lower than this number.
        penalty: Choose between 'mmd' and 'cts-mmd'. 'mmd' calculates MMD penalty between each pair of batches. 'cts-mmd' calculates MMD penalty between each pair of batches in each cell type.
        niche_edge: whether to use niches for modeling spatial location. Refer to https://doi.org/10.1038/s41587-024-02193-4 for further details.
        subsample: subsample the dataset to calculate MMD penalty if the dataset contains too many cells.
        epoch: number of iteration steps.
        batch_size: input cells in each batch.
        batch_key: key of batch information in adata.obs.
        spatial_key: key of spatial location in adata.obsm.
        ct_key: key of cell type information in adata.obs.
    
    Output:
        adata: result adata with velocity and latent time estimate.
        spvelo model and adata saved at save_dir+f'models/{save_name}_{method_name}.pkl'.
    '''
    from ._model import SPVELO
    from ._utils import preprocess_data, save_obj, load_obj, setup_seed, add_velovi_outputs_to_adata
    
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    adata.X = np.asarray(adata.X)
    if adata.shape[1]>2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key=batch_key)
    if filter_uninfo != None:
        adata = preprocess_data(adata, filter_on_r2=True, r2_thresh=filter_uninfo)
    else:
        adata = preprocess_data(adata, filter_on_r2=False)
    adata.layers['index'] = np.zeros(adata.X.shape)
    adata.layers['index'][:,0] = [i for i in range(adata.X.shape[0])]
    SPVELO.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu", index_data="index", batch_data=batch_key, label_data=ct_key, spatial_data=spatial_key)
    if niche_edge == True:
        spnn_idx, spnn_weight = new_BatchKNN(adata, batch_key=batch_key, nn_key=spatial_key, k=8)
        adata = aggr_niche(adata, spnn_idx, spnn_weight)
        pca_lst = []
        for b in list(dict.fromkeys(adata.obs[batch_key])):
            bdata = adata[adata.obs[batch_key]==b]
            bdata.X = bdata.layers['aggr']
            sc.pp.pca(bdata, n_comps=30)
            pca_lst.append(list(bdata.obsm['X_pca']))
        adata.obsm['X_niche_pca'] = np.concatenate(pca_lst, axis=0)
    
    if add_mnn == True:
        get_mnn(adata, batch_key=batch_key, n_bnn_neighbors=mnn_neighbors, is_ot=True, use_rep='X_pca')
        
    vae = SPVELO(adata, penalty=penalty, add_mnn=add_mnn, niche_edge=niche_edge, subsample=subsample, spatial_neighbors=spatial_neighbors)
    vae.train(batch_size=batch_size, max_epochs=epoch)
    add_velovi_outputs_to_adata(adata, vae)
    adata.write_h5ad(save_dir+f'adatas/{save_name}_{method_name}.h5ad')
    save_obj(vae, save_dir+f'models/{save_name}_{method_name}.pkl')
    
    return adata
    
def run_scv_stc(adata, save_name, save_dir, method_name='stc', batch_key='batch', ct_key='cluster_annotations'):
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    adata.X = np.asarray(adata.X)
    if adata.shape[1]>2000:
        scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode='stochastic')
    adata.write_h5ad(save_dir+'adatas/{}_{}.h5ad'.format(save_name, method_name))
    
def run_scv_dyn(adata, save_name, save_dir, method_name='dyn', batch_key='batch', ct_key='cluster_annotations'):
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    adata.X = np.asarray(adata.X)
    if adata.shape[1]>2000:
        scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata)
    valid_columns = np.where(~np.isnan(adata.layers['fit_t']).any(axis=0))[0]
    adata = adata[:, valid_columns]
    scv.tl.velocity(adata, mode='dynamical')
    valid_columns = np.where(~np.isnan(adata.layers['velocity']).any(axis=0))[0]
    adata = adata[:, valid_columns]
    adata.write_h5ad(save_dir+'adatas/{}_{}.h5ad'.format(save_name, method_name))

def run_vi(adata, save_name, save_dir, method_name='vi', batch_key='batch', ct_key='cluster_annotations', filter_genes=True):
    from velovi import preprocess_data, VELOVI
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    adata.X = np.asarray(adata.X)
    if adata.shape[1]>2000:
        scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata = preprocess_data(adata, filter_on_r2=filter_genes)
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train()
    add_velovi_outputs_to_adata(adata, vae)
    adata.write_h5ad(save_dir+'adatas/{}_{}.h5ad'.format(save_name, method_name))
    save_obj(vae, save_dir+'models/{}_{}.pkl'.format(save_name, method_name))

def run_std_ltv(adata, save_name, save_dir, method_name='std_ltv', epochs=50, batch_size=100, batch_key='batch', ct_key='cluster_annotations'):
    import latentvelo as ltv
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    adata.X = np.asarray(adata.X)
    adata.layers['spliced'] = adata.layers['spliced_counts'].copy()
    adata.layers['unspliced'] = adata.layers['unspliced_counts'].copy()
    adata = ltv.utils.standard_clean_recipe(adata, spliced_key='spliced', unspliced_key='unspliced',
                                batch_key=batch_key, celltype_key=ct_key)
    model = ltv.models.VAE(observed = adata.shape[1], latent_dim = 20, zr_dim = 1, h_dim = 2, batch_correction = True, 
                                batches = len(list(set(adata.obs[batch_key]))))
    epochs, val_ae, val_traj = ltv.train_vae(model, adata, batch_size=batch_size, epochs=epochs, name='latentvelo')
    latent_adata, adata = ltv.output_results(model, adata, gene_velocity = True, decoded=True)
    if 'initial_size' in adata.obs.keys():
        adata.obs = adata.obs.drop('initial_size', axis=1)
    adata.layers['velocity'] = adata.layers['velo']
    adata.write_h5ad(save_dir+'adatas/{}_std_ltv.h5ad'.format(save_name))
    if 'initial_size' in latent_adata.obs.keys():
        latent_adata.obs = latent_adata.obs.drop('initial_size', axis=1)
    latent_adata.write_h5ad(save_dir+'adatas/{}_std_ltv_ld.h5ad'.format(save_name))
    
def run_annot_ltv(adata, save_name, save_dir, method_name='annot_ltv', epochs=50, batch_size=100, batch_key='batch', ct_key='cluster_annotations'):
    import latentvelo as ltv
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    adata.X = np.asarray(adata.X)
    adata.layers['spliced'] = adata.layers['spliced_counts'].copy()
    adata.layers['unspliced'] = adata.layers['unspliced_counts'].copy()
    adata = ltv.utils.anvi_clean_recipe(adata, spliced_key='spliced', unspliced_key='unspliced',
                                batch_key=batch_key, celltype_key=ct_key)
    model = ltv.models.AnnotVAE(observed = adata.shape[1], latent_dim = 20, zr_dim = 1, h_dim = 2, 
                                celltypes = len(list(set(adata.obs[ct_key]))), batch_correction = True, 
                                batches = len(list(set(adata.obs[batch_key]))))
    epochs, val_ae, val_traj = ltv.train_anvi(model, adata, batch_size=batch_size, epochs=epochs, name='latentvelo')
    latent_adata, adata = ltv.output_results(model, adata, gene_velocity = True, decoded=True)
    if 'initial_size' in adata.obs.keys():
        adata.obs = adata.obs.drop('initial_size', axis=1)
    adata.layers['velocity'] = adata.layers['velo']
    adata.write_h5ad(save_dir+'adatas/{}_annot_ltv.h5ad'.format(save_name))
    if 'initial_size' in latent_adata.obs.keys():
        latent_adata.obs = latent_adata.obs.drop('initial_size', axis=1)
    latent_adata.write_h5ad(save_dir+'adatas/{}_annot_ltv_ld.h5ad'.format(save_name))
