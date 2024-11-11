from .eval_util import *
import scvelo as scv
from spVelo.spVelo._utils import *

def benchmark_mnn(adatas_dict, cluster_edges, wrong_edges, save_name, save_dir, n_neighbors=30, minmaxscale=False, batch_key='batch', spatial_key='spatial', ct_key='cluster_annotations', velocity_key='velocity', save=True):
    result_table_mnn = {'confidence':{},'transition':{},'direction':{}}
    for method in adatas_dict.keys():
        adata = adatas_dict[method].copy()
        get_mnn(adata, batch_key='batch', n_bnn_neighbors=n_neighbors, is_ot=False, use_rep='X_pca')
        assert adata.uns['neighbors']['params']['method'] == 'mnn', "Wrong mnn setting! Rerun with is_ot=False"
        #filter cells with all distances=0
        print('filtering {} cells with all distances==0'.format(((adata.obsp['mnn_distances'] > 0).sum(1).A1==0).sum()))
        adata = adata[(adata.obsp['mnn_distances'] > 0).sum(1).A1!=0,:]   
        
        adata.obsp['connectivities'] = adata.obsp["mnn_connectivities"]
        adata.obsp['distances'] = adata.obsp["mnn_distances"]
        scv.tl.velocity_graph(adata)
        scv.tl.velocity_confidence(adata, vkey='velocity')
        adata.uns['velocity_trans_prob'] = scv.utils.get_transition_matrix(adata)
        assert (adata.obsp['connectivities'].todense() == adata.obsp['mnn_connectivities'].todense()).all(), "scvelo automatically recalculated neighbors"
        sc.tl.umap(adata)
        scv.pl.velocity_embedding_stream(adata, basis='X_umap',color=ct_key,
                                    save=save_dir+'plots/'+save_name+'_{}_mnn.png'.format(method)) 
        scv.pl.velocity_embedding_stream(adata, basis='X_pca',color=ct_key)
        
        pos_mnn_scores, comb_mnn_scores = evaluate_mnn(adata, cluster_edges, wrong_edges, k_cluster=ct_key, batch_key=batch_key, spatial_key=None, k_velocity=velocity_key, verbose=False)
        if save==True:
            save_obj(pos_mnn_scores, save_dir+'scores/{}_{}_pos_mnn.pkl'.format(save_name, method))
            save_obj(comb_mnn_scores, save_dir+'scores/{}_{}_comb_mnn.pkl'.format(save_name, method))
        
        for score in comb_mnn_scores.keys():
            result_table_mnn[score][method] = comb_mnn_scores[score]
        
    df_mnn = pd.DataFrame(result_table_mnn)
    if minmaxscale==True:
        df_mnn = pd.DataFrame(
                MinMaxScaler().fit_transform(df_mnn),
                columns=df_mnn.columns,
                index=df_mnn.index,
            )
    return df_mnn
        
def benchmark_perbatch_sp(adatas_dict, cluster_edges, wrong_edges, save_name, save_dir, batch_name, n_neighbors=30, minmaxscale=False, batch_key=None, spatial_key='spatial', ct_key='cluster_annotations', velocity_key='velocity', save=True):
    result_table = {'confidence':{}, 'transition':{}, 'direction':{}}
    for method in adatas_dict.keys():
        adata = adatas_dict[method].copy()
        if batch_key == None: #for benchmarking perbatch methods
            get_spatial_neighbor(adata, spatial_key=spatial_key, batch_key=None, n_neighs=n_neighbors)
            adata.uns['neighbors'] = adata.uns['spatial_neighbors'].copy()
            adata.obsp['distances'] = adata.obsp['spatial_distances']
            adata.obsp['connectivities'] = adata.obsp['spatial_connectivities']
            scv.tl.velocity_graph(adata)
            scv.tl.velocity_confidence(adata, vkey='velocity')
            adata.uns['velocity_trans_prob'] = scv.utils.get_transition_matrix(adata)
            assert (adata.obsp['connectivities'].todense() == adata.obsp['spatial_connectivities'].todense()).all(), "scvelo automatically recalculated neighbors, set adata.obsp['connectivities'].todense() = adata.obsp['spatial_connectivities'].todense()"
            scv.pl.velocity_embedding_stream(adata, basis=spatial_key,color=ct_key, X=adata.obsm[spatial_key], 
                                    save=save_dir+'plots/'+save_name+'_{}_sp.png'.format(method))
            
            pos_scores, comb_scores = evaluate(adata, cluster_edges, wrong_edges, x_emb=spatial_key, k_cluster=ct_key, batch_key=batch_key, spatial_key=spatial_key, k_velocity=velocity_key, verbose=False)
            
        elif batch_key != None: #for benchmarking integrated methods on perbatch experiments, evaluate only for one batch
            bdata = adata[np.where(adata.obs[batch_key]==batch_name)]
            #get perbatch confidence & transprob & plots
            get_spatial_neighbor(bdata, spatial_key=spatial_key, batch_key=None, n_neighs=n_neighbors)
            bdata.uns['neighbors'] = bdata.uns['spatial_neighbors'].copy()
            bdata.obsp['distances'] = bdata.obsp['spatial_distances']
            bdata.obsp['connectivities'] = bdata.obsp['spatial_connectivities']
            scv.tl.velocity_graph(bdata)
            scv.tl.velocity_confidence(bdata, vkey='velocity')
            bdata.uns['velocity_trans_prob'] = scv.utils.get_transition_matrix(bdata)
            assert (bdata.obsp['connectivities'].todense() == bdata.obsp['spatial_connectivities'].todense()).all(), "scvelo automatically recalculated neighbors"
            scv.pl.velocity_embedding_stream(bdata, basis=spatial_key,color=ct_key, X=bdata.obsm[spatial_key], 
                                    save=save_dir+'plots/'+save_name+'_{}_sp.png'.format(method))
            
            pos_scores, comb_scores = evaluate(bdata, cluster_edges, wrong_edges, x_emb=spatial_key, k_cluster=ct_key, batch_key=None, spatial_key=spatial_key, k_velocity=velocity_key, verbose=False)
                
        if save==True:
            save_obj(pos_scores, save_dir+'scores/{}_{}_pos_sp.pkl'.format(save_name, method))
            save_obj(comb_scores, save_dir+'scores/{}_{}_comb_sp.pkl'.format(save_name, method))
        
        for score in comb_scores.keys():
            result_table[score][method] = comb_scores[score]
        
    df = pd.DataFrame(result_table)
    if minmaxscale==True:
        df = pd.DataFrame(
                MinMaxScaler().fit_transform(df),
                columns=df.columns,
                index=df.index,
            )
    return df


def benchmark_perbatch_expr(adatas_dict, cluster_edges, wrong_edges, save_name, save_dir, batch_name, n_neighbors=30, minmaxscale=True, batch_key=None, ct_key='cluster_annotations', velocity_key='velocity', save=True): #no spatial parameter
    result_table = {'confidence':{},'transition':{},'direction':{}}
    for method in adatas_dict.keys():
        adata = adatas_dict[method].copy()
        if batch_key == None: #for benchmarking perbatch methods
            scv.pp.neighbors(adata, n_neighbors=n_neighbors)
            scv.tl.velocity_graph(adata)
            scv.tl.velocity_confidence(adata, vkey='velocity')
            adata.uns['velocity_trans_prob'] = scv.utils.get_transition_matrix(adata)
            scv.pl.velocity_embedding_stream(adata, basis='X_umap',color=ct_key,
                                    save=save_dir+'plots/'+save_name+'_{}_expr.png'.format(method)) 
            scv.pl.velocity_embedding_stream(adata, basis='X_pca',color=ct_key)
            pos_scores, comb_scores = evaluate(adata, cluster_edges, wrong_edges, k_cluster=ct_key, batch_key=batch_key, spatial_key=None, k_velocity=velocity_key, verbose=False)
        elif batch_key != None: #for benchmarking integrated methods on perbatch experiments, evaluate only for one batch
            bdata = adata[np.where(adata.obs[batch_key]==batch_name)]
            #get perbatch confidence & transprob & plots
            scv.pp.neighbors(bdata, n_neighbors=n_neighbors)
            scv.tl.velocity_graph(bdata)
            scv.tl.velocity_confidence(bdata, vkey='velocity')
            bdata.uns['velocity_trans_prob'] = scv.utils.get_transition_matrix(bdata)
            sc.tl.umap(bdata) #need to recalculate X_umap due to recalculation of perbatch neighbors
            scv.pl.velocity_embedding_stream(bdata, basis='X_umap',color=ct_key,
                                    save=save_dir+'plots/'+save_name+'_{}_expr.png'.format(method)) 
            scv.pl.velocity_embedding_stream(bdata, basis='X_pca',color=ct_key)
            pos_scores, comb_scores = evaluate(bdata, cluster_edges, wrong_edges, k_cluster=ct_key, batch_key=None, spatial_key=None, k_velocity=velocity_key, verbose=False)
                
        if save==True:
            save_obj(pos_scores, save_dir+'scores/{}_{}_pos_expr.pkl'.format(save_name, method))
            save_obj(comb_scores, save_dir+'scores/{}_{}_comb_expr.pkl'.format(save_name, method))
        
        for score in comb_scores.keys():
            result_table[score][method] = comb_scores[score]
        
    df = pd.DataFrame(result_table)
    if minmaxscale==True:
        df = pd.DataFrame(
                MinMaxScaler().fit_transform(df),
                columns=df.columns,
                index=df.index,
            )
    return df
        
      
