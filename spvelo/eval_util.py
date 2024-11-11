# -*- coding: utf-8 -*-
"""Evaluation utility functions.

This module contains util functions for computing evaluation scores.

"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import squidpy as sq
from itertools import combinations, permutations
from collections import OrderedDict
import scipy
from scipy.sparse import csr_matrix, bmat
from sklearn.metrics import pairwise_distances
import ot
import torch
import scanpy as sc
from scvelo.preprocessing.neighbors import _get_rep, _set_pca, get_duplicate_cells#, _set_neighbors_data
from scvelo import logging as logg
from scanpy.neighbors import _get_indices_distances_from_dense_matrix
from velovgi.preprocessing.batch_network import *
from anndata import AnnData

def get_spatial_neighbor(adata, spatial_key="spatial", batch_key=None, n_neighs=30):
    if batch_key==None:
        sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, n_neighs=n_neighs)
    else:
        lst = list(adata.obs[batch_key])
        batch_list = list(OrderedDict.fromkeys(lst))
        if len(batch_list)==1:
            sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, n_neighs=n_neighs)
        else:
            adatas = {}
            mtx = np.zeros((adata.shape[0],adata.shape[0]))
            mtx2 = np.zeros((adata.shape[0],adata.shape[0]))
            row_idx = 0
            for batch in batch_list:
                bdata = adata[np.where(adata.obs[batch_key]==batch)].copy()
                sq.gr.spatial_neighbors(bdata, spatial_key=spatial_key, n_neighs=n_neighs)
                mtx[row_idx:row_idx+bdata.shape[0], row_idx:row_idx+bdata.shape[0]] = bdata.obsp['spatial_connectivities'].todense()
                mtx2[row_idx:row_idx+bdata.shape[0], row_idx:row_idx+bdata.shape[0]] = bdata.obsp['spatial_distances'].todense()
                row_idx += bdata.shape[0]
            adata.obsp['spatial_connectivities'] = csr_matrix(mtx)
            adata.obsp['spatial_distances'] = csr_matrix(mtx2)
            #add .uns['spatial_neighbors']
            adata.uns['spatial_neighbors'] = {'connectivities_key': 'spatial_connectivities',
                                              'distances_key': 'spatial_distances',
                                              'params': {'n_neighbors': 30,
                                                         'coord_type': 'generic',
                                                         'radius': None,
                                                         'transform': None}}

def _set_neighbors_data(
    adata: AnnData,
    neighbors,
    n_neighbors: int,
    method: str,
    metric: str,
    n_pcs: int,
    use_rep: str,
):
    #from scvelo.pp.neighbors, change to fit mnn
    adata.uns["neighbors"] = {}
    # TODO: Remove except statement. `n_obs` x `n_obs` arrays need to be written to
    # `AnnData.obsp`.
    try:
        adata.obsp["mnn_distances"] = neighbors.distances
        adata.obsp["mnn_connectivities"] = neighbors.connectivities
        adata.uns["neighbors"]["connectivities_key"] = "mnn_connectivities"
        adata.uns["neighbors"]["distances_key"] = "mnn_distances"
    except ValueError as e:
        logg.warning(f"Could not write neighbors to `AnnData.obsp`: {e}")
        adata.uns["neighbors"]["distances"] = neighbors.distances
        adata.uns["neighbors"]["connectivities"] = neighbors.connectivities

    if hasattr(neighbors, "knn_indices"):
        adata.uns["neighbors"]["indices"] = neighbors.knn_indices
    adata.uns["neighbors"]["params"] = {
        "n_neighbors": n_neighbors,
        "method": method,
        "metric": metric,
        "n_pcs": n_pcs,
        "use_rep": use_rep,
    }

def get_mnn(adata, batch_key="batch", n_bnn_neighbors=30, batch_pair_list=None, is_ot=True,
             ratio_bnn = None, n_pcs=None, use_rep="X_pca", use_highly_variable=False, metric="euclidean"):
    #get mutual nearest neighbors, from velovgi.
    max_n_bnn_neighbors = 50
    use_rep = _get_rep(adata=adata, use_rep=use_rep, n_pcs=n_pcs)
    if use_rep == "X_pca":
        _set_pca(adata=adata, n_pcs=n_pcs, use_highly_variable=use_highly_variable)

        n_duplicate_cells = len(get_duplicate_cells(adata))
        if n_duplicate_cells > 0:
            logg.warn(
                f"You seem to have {n_duplicate_cells} duplicate cells in your data.",
                "Consider removing these via pp.remove_duplicate_cells.",
            )
    logg.info(f"use_rep : {use_rep}", r=True)
    # X = adata.X if use_rep == "X" else adata.obsm[use_rep]

    batch_list = list(adata.obs[batch_key].cat.categories)
    adata_list = [adata[adata.obs[batch_key]==batch].copy() for batch in batch_list]

    m = len(batch_list)
    connectivities_list = [[None for j in range(m)]for i in range(m)] 
    distances_list = [[None for j in range(m)]for i in range(m)]

    if batch_pair_list == None:
        batch_pair_list = []
        l = len(batch_list)
        for i in range(l):
            for j in range(i + 1, l):
                batch_pair_list.append([batch_list[i], batch_list[j]])
    logg.info(f"batch_pair_list : {batch_pair_list}", r=True)
    for batch_pair in batch_pair_list:
        batch1_index, batch2_index = batch_list.index(batch_pair[0]), batch_list.index(batch_pair[1])
        adata1, adata2 = adata_list[batch1_index], adata_list[batch2_index]
        X = adata1.X if use_rep == "X" else adata1.obsm[use_rep]
        Y = adata2.X if use_rep == "X" else adata2.obsm[use_rep]
        if not (ratio_bnn == None):
            n1 = adata1.shape[0]
            n2 = adata2.shape[0]
            n_bnn_neighbors = int((n1 + n2)/2 *  ratio_bnn)
            n_bnn_neighbors = min(n_bnn_neighbors, max_n_bnn_neighbors)
            n_bnn_neighbors = max(n_bnn_neighbors, 1)
            # actual_max_n_bnn_neighbors = max(actual_max_n_bnn_neighbors, n_bnn_neighbors)
            logg.info(f"pair {batch_pair} n_bnn_neighbors: {n_bnn_neighbors}")
        if (n_bnn_neighbors > adata1.shape[0]) or (n_bnn_neighbors > adata2.shape[0]):
            k = min(adata1.shape[0], adata2.shape[0])
            logg.info(f"pair {batch_pair} cells not enough, k={k}", r=True)
        else:
            k = n_bnn_neighbors
        distances = pairwise_distances(X,Y)
        if is_ot == True:
            a, b = np.ones((adata1.shape[0],)) / adata1.shape[0], np.ones((adata2.shape[0],)) / adata2.shape[0]
            connectivities = ot.emd(a, b, distances)
            filtered_connectivities = filter_M(connectivities, k, largest=True)
            mnn_distances = np.where(filtered_connectivities>0, distances, 0)
            mnn_connectivities = get_normed_mnn_connectivities(filtered_connectivities)
        else:
            mnn_distances = filter_M(distances, k, largest=False)
            mnn_connectivities = get_mnn_connectivities(mnn_distances)

        connectivities_list[batch1_index][batch2_index] =  mnn_connectivities
        connectivities_list[batch2_index][batch1_index] = mnn_connectivities.T
        distances_list[batch1_index][batch2_index] = mnn_distances
        distances_list[batch2_index][batch1_index] = mnn_distances.T

    adata_concat = sc.concat(adata_list)
    adata_concat.obsp["mnn_connectivities"] = bmat(connectivities_list).A
    adata_concat.obsp["mnn_distances"] = bmat(distances_list).A
    adata_concat = adata_concat[adata.obs.index]

    class Neighbor():
        def __init__(self, indices, distances, connectivities):
            self.knn_indices = indices
            self.distances = distances
            self.connectivities = connectivities
        
    n_neighbors = n_bnn_neighbors
    distance = adata_concat.obsp["mnn_distances"]
    distance = np.where(distance>0, distance, np.inf)
    np.fill_diagonal(distance, 0)
    indices, nn_distances = _get_indices_distances_from_dense_matrix(distance, n_neighbors)
    indices = np.where(nn_distances!=np.inf, indices, -1)

    distances = csr_matrix(adata_concat.obsp["mnn_distances"])
    connectivities =  csr_matrix(adata_concat.obsp["mnn_connectivities"])
    neighbors = Neighbor(indices, distances, connectivities)
    if is_ot == True:
        mnn_name = "mnn_ot"
    else:
        mnn_name = "mnn"
    _set_neighbors_data(adata,
                        neighbors,
                        n_neighbors=n_neighbors,
                        method=mnn_name,
                        metric=metric,
                        n_pcs=n_pcs,
                        use_rep=use_rep,
                        )
        
def summary_scores(all_scores, cluster_edges=None):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: Group-wise aggregation scores.
        float: score aggregated on all samples
        
    """
    #sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s }
    #include empty key in sep_scores
    sep_scores = {}
    if cluster_edges == None:
        cluster_edges = all_scores.keys()
    for k in cluster_edges:
        if all_scores[k]!=[]:
            sep_scores[k] = np.mean(all_scores[k])
        else:
            sep_scores[k] = 0
    overal_agg = np.mean([s for k, s in sep_scores.items()])
    return sep_scores, overal_agg

def summary_scores_including_wrong(all_scores, cluster_edges, wrong_edges):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: Group-wise aggregation scores.
        float: score aggregated on all samples
        
    """
    #sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s }
    #include empty key in sep_scores
    sep_scores = {}
    for k in all_scores.keys():
        if all_scores[k]!=[]:
            sep_scores[k] = np.mean(all_scores[k])
        else:
            sep_scores[k] = 0
    
    correct_scores = [s for k, s in sep_scores.items() if k in cluster_edges]
    wrong_scores = [-s for k, s in sep_scores.items() if k in wrong_edges]
    new_scores = correct_scores + wrong_scores
    overal_agg = np.mean(new_scores)
    return overal_agg


def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        k_cluster (str): Cluster key in adata.obs dataframe

    Returns:
        list: Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]


def cross_boundary_scvelo_probs(adata, k_cluster, cluster_edges, k_trans_g, batch_key=None, spatial_key=None, return_raw=True, n_neighs=30):
    """Compute Cross-Boundary Confidence Score (A->B), i.e. transition score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        k_trans_g (str): key to the transition graph computed using velocity.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    
    scores = {}
    all_scores = {}
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        if spatial_key==None:
            connectivity = np.array(adata.obsp['connectivities'][sel].todense())
        else:
            connectivity = np.array(adata.obsp['spatial_connectivities'][sel].todense())
        idx = []
        type_score = []
        for i in range(connectivity.shape[0]):
            #idx.append(np.where(connectivity[i]==1)[0])
            idx = np.where(connectivity[i]!=0)[0].astype(int)
            nodes = keep_type(adata, idx, target=v, k_cluster=k_cluster)
            if len(nodes)>0:
                type_score.append(adata.uns[k_trans_g][sel].toarray()[i, nodes].mean())
        #nbs = np.vstack(tuple(idx))
        #boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        #type_score = [trans_probs.toarray()[:, nodes].mean() 
        #              for trans_probs, nodes in zip(adata.uns[k_trans_g][sel], boundary_nodes) 
        #              if len(nodes) > 0]
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    if return_raw:
        return all_scores    
    return scores, np.mean([sc for sc in scores.values()])

def cross_boundary_coh(adata, k_cluster, k_velocity, cluster_edges, batch_key=None, spatial_key=None, return_raw=True, n_neighs=30):
    """Cross-Boundary Velocity Coherence Score (A->B).
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    velocities = adata.layers[k_velocity]  
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        v_us = velocities[sel]
        if spatial_key==None:
            connectivity = np.array(adata.obsp['connectivities'][sel].todense())
        else:
            connectivity = np.array(adata.obsp['spatial_connectivities'][sel].todense())
        #idx = []
        type_score = []
        for i in range(connectivity.shape[0]):
            idx = np.where(connectivity[i]!=0)[0].astype(int)
            nodes = keep_type(adata, idx, target=v, k_cluster=k_cluster)
            if len(nodes)>0:
                type_score.append(cosine_similarity(v_us[[i]], velocities[nodes]).mean())
            #idx.append(np.where(connectivity[i]==1)[0])
        #nbs = np.vstack(tuple(idx))
        #boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        
        #type_score = [cosine_similarity(v_us[[ith]], velocities[nodes]).mean()
        #              for ith, nodes in enumerate(boundary_nodes) 
        #              if len(nodes) > 0]
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

def cross_boundary_confidence(adata, k_cluster, k_confidence, cluster_edges, batch_key=None, spatial_key=None, return_raw=True, n_neighs=30):
    """Cross-Boundary Velocity Confidence Score (A->B).
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        if spatial_key==None:
            connectivity = np.array(adata.obsp['connectivities'][sel].todense())
        else:
            connectivity = np.array(adata.obsp['spatial_connectivities'][sel].todense())
        #idx = []
        type_score = []
        for i in range(connectivity.shape[0]):
            idx = np.where(connectivity[i]!=0)[0].astype(int)
            nodes = keep_type(adata, idx, target=v, k_cluster=k_cluster)
            if len(nodes)>0:
                #type_score.append(cosine_similarity(v_us[[i]], velocities[nodes]).mean())
                type_score.append(np.mean(adata.obs[k_confidence][sel].values.tolist()))
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

def in_cluster_scvelo_coh(adata, k_cluster, k_confidence, batch_key=None, spatial_key=None, return_raw=True):
    """In-Cluster Confidence Score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_confidence (str): key to the column of cell velocity confidence in adata.obs.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
    
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
        
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        type_score = adata.obs[k_confidence][sel].values.tolist() 
        scores[cat] = np.mean(type_score)
        all_scores[cat] = type_score
        
    if return_raw:
        return all_scores
    
    return scores, np.mean([s for _, s in scores.items()])

def cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, batch_key=None, spatial_key=None, return_raw=True, x_emb="X_umap", n_neighs=30):
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        x_emb (str): key to x embedding for visualization.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    if x_emb != None:
        x_embedding = adata.obsm[x_emb]
        if x_emb == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        elif x_emb == "X_pca":
            v_emb = adata.obsm['{}_pca'.format(k_velocity)]
        elif x_emb == "spatial":
            v_emb = adata.obsm['{}_spatial'.format(k_velocity)]
        #else:
        #    v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.todense()
        x_embedding = np.asarray(adata.X)
        v_emb = adata.layers['velocity']

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        x_points = x_embedding[sel]
        x_velocities = v_emb[sel]
        if spatial_key==None:
            connectivity = np.array(adata.obsp['connectivities'][sel].todense())
        else:
            connectivity = np.array(adata.obsp['spatial_connectivities'][sel].todense())
        type_score = []
        for i in range(connectivity.shape[0]):
            idx = np.where(connectivity[i]!=0)[0].astype(int)
            nodes = keep_type(adata, idx, target=v, k_cluster=k_cluster)
            if len(nodes)>0:
                x_pos = x_points[i]
                x_vel = x_velocities[i]
                position_dif = x_embedding[nodes] - x_pos
                dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
                type_score.append(np.mean(dir_scores))
        
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])


def inner_cluster_coh(adata, k_cluster, k_velocity, batch_key=None, spatial_key=None, return_raw=True, n_neighs=30):
    """In-cluster Coherence Score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    velocities = adata.layers[k_velocity]
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        cat_vels = velocities[sel]
        if spatial_key==None:
            connectivity = np.array(adata.obsp['connectivities'][sel].todense())
        else:
            connectivity = np.array(adata.obsp['spatial_connectivities'][sel].todense())
        cat_score = []
        for i in range(connectivity.shape[0]):
            idx = np.where(connectivity[i]!=0)[0].astype(int)
            nodes = keep_type(adata, idx, target=cat, k_cluster=k_cluster)
            if len(nodes)>0:
                cat_score.append(cosine_similarity(cat_vels[[i]], velocities[nodes]).mean())
                #idx.append(np.where(connectivity[i]==1)[0])
            #nbs = np.vstack(tuple(idx))
        #same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, k_cluster), nbs)
        
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])


def evaluate(adata, cluster_edges, wrong_edges, k_cluster, k_velocity, x_emb="X_pca", batch_key=None, spatial_key="spatial", verbose=True, n_neighs=30):
    """Evaluate velocity estimation results using 5 metrics.
    
    Args:
        adata (Anndata): Anndata object.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        x_emb (str): key to x embedding for visualization.
        
    Returns:
        dict: aggregated metric scores.
    
    """
    
    #reject wrong transitory direction
    celltypes = list(set(adata.obs[k_cluster]))
    all_edges = list(permutations(celltypes, 2)) + [(i,i) for i in celltypes]
    
    conf = cross_boundary_confidence(adata, k_cluster, "{}_confidence".format(k_velocity), all_edges, batch_key, spatial_key, True, n_neighs)
    trans_probs_sp = cross_boundary_scvelo_probs(adata, k_cluster, all_edges, "{}_trans_prob".format(k_velocity), batch_key, spatial_key, True, n_neighs)
    if spatial_key == None:
        crs_bdr_crc_sp_pca = cross_boundary_correctness(adata, k_cluster, k_velocity, all_edges, batch_key, spatial_key, True, x_emb, n_neighs)
    else:
        crs_bdr_crc_sp_pca = cross_boundary_correctness(adata, k_cluster, k_velocity, all_edges, batch_key, spatial_key, True, "spatial", n_neighs)
    
    pos_scores = {
        "confidence": summary_scores(conf, cluster_edges)[0],
        "transition": summary_scores(trans_probs_sp)[0],
        "direction": summary_scores(crs_bdr_crc_sp_pca)[0],
    }
    
    combined_scores = {
        "confidence": summary_scores(conf, cluster_edges)[1],
        "transition": summary_scores_including_wrong(trans_probs_sp, cluster_edges, wrong_edges),
        "direction": summary_scores_including_wrong(crs_bdr_crc_sp_pca, cluster_edges, wrong_edges),
    }
    
    return pos_scores, combined_scores

def evaluate_mnn(adata, cluster_edges, wrong_edges, k_cluster, k_velocity, x_emb="X_pca", batch_key="batch", spatial_key=None, verbose=True, n_neighs=30):
    """Evaluate velocity estimation results using 5 metrics.
    
    Args:
        adata (Anndata): Anndata object.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        x_emb (str): key to x embedding for visualization.
        
    Returns:
        dict: aggregated metric scores.
    
    """
    #reject wrong transitory direction
    celltypes = list(set(adata.obs[k_cluster]))
    all_edges = list(permutations(celltypes, 2)) + [(i,i) for i in celltypes]
    
    conf = cross_boundary_confidence(adata, k_cluster, "{}_confidence".format(k_velocity), all_edges, batch_key, spatial_key, True, n_neighs)
    trans_probs_sp = cross_boundary_scvelo_probs(adata, k_cluster, all_edges, "{}_trans_prob".format(k_velocity), batch_key, spatial_key, True, n_neighs)
    crs_bdr_crc_sp_pca = cross_boundary_correctness(adata, k_cluster, k_velocity, all_edges, batch_key, spatial_key, True, x_emb, n_neighs)
    
    pos_scores = {
        "confidence": summary_scores(conf)[0],
        "transition": summary_scores(trans_probs_sp)[0],
        "direction": summary_scores(crs_bdr_crc_sp_pca)[0],
    }
    
    combined_scores = {
        "confidence": summary_scores(conf, cluster_edges)[1],
        "transition": summary_scores_including_wrong(trans_probs_sp, cluster_edges, wrong_edges),
        "direction": summary_scores_including_wrong(crs_bdr_crc_sp_pca, cluster_edges, wrong_edges),
    }
    
    return pos_scores, combined_scores
