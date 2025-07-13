import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scCube
from scCube import scCube
from scCube.visualization import *
from scCube.utils import *
import torch
import random
import pytorch_lightning
import pickle
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

# preprocess
adata = scv.datasets.pancreas()
adata.layers['spliced_counts'] = adata.layers['spliced'].copy()
adata.layers['unspliced_counts'] = adata.layers['unspliced'].copy()
adata.layers['matrix'] = (adata.layers['spliced_counts']+adata.layers['unspliced_counts'])
adata.X = adata.layers['matrix'].copy()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, subset_highly_variable=False, enforce=True)

# simulation using scCube
adata.X = adata.X.todense()
sc_data = pd.DataFrame(adata.X.transpose(), index=adata.var_names, columns=adata.obs_names)
sc_meta = pd.DataFrame(adata.obs.clusters)
sc_meta.columns = ['Cell_type']
sc_meta['Cell'] = adata.obs_names

adatas = []
for seed in range(3):
    setup_seed(seed)
    model = scCube()
    generate_sc_data_new, generate_sc_meta_new = model.generate_pattern_random(generate_sc_data=sc_data,generate_sc_meta=sc_meta,set_seed=True,seed=seed,
                                                                                spatial_cell_type=None,spatial_dim=2,spatial_size=50,delta=25,lamda=0.75,)
    plot_spatial_pattern_scatter(obj=generate_sc_meta_new,figwidth=2.5,figheight=2.5,dim=2,x="point_x",y="point_y",label='Cell_type',colormap='turbo',size=10,alpha=1)
    plt.show()
    
    spliced_data = pd.DataFrame(adata.layers['spliced'].todense().transpose(), index=adata.var_names, columns=adata.obs_names)
    st_spliced_data, st_spliced_meta, st_spliced_index = model.generate_spot_data_random(generate_sc_data=spliced_data,generate_sc_meta=generate_sc_meta_new,platform='ST',gene_type='hvg',min_cell=1,n_gene=2000,n_cell=1)
    unspliced_data = pd.DataFrame(adata.layers['unspliced'].todense().transpose(), index=adata.var_names, columns=adata.obs_names)
    st_unspliced_data, st_unspliced_meta, st_unspliced_index = model.generate_spot_data_random(generate_sc_data=unspliced_data,generate_sc_meta=generate_sc_meta_new,platform='ST',gene_type='hvg',min_cell=1,n_gene=2000,n_cell=1)
    df = st_spliced_index.drop_duplicates(subset=['spot'], keep='first')
    df = df.reset_index(drop=True)
    df.index = df['spot']
    df = df.loc[st_spliced_data.transpose().index,:]
    
    sim_data = AnnData(st_spliced_data.transpose(), obs=df)
    sim_data.layers['spliced'] = st_spliced_data.transpose()
    sim_data.layers['unspliced'] = st_unspliced_data.transpose()
    sim_data.obs['spot_x'] = sim_data.obs['spot_x'].astype('float')
    sim_data.obs['spot_y'] = sim_data.obs['spot_y'].astype('float')
    sim_data.obsm['spatial'] = np.array(sim_data.obs.loc[:,['spot_x','spot_y']])
    
    sim_data.obs["batch"] = str(seed+1)
    sc.pp.normalize_total(sim_data, target_sum=1e4)
    sc.pp.log1p(sim_data)
    sc.tl.pca(sim_data, n_comps=30)
    sc.pp.neighbors(sim_data, n_neighbors=30)
    sc.tl.umap(sim_data)
    scv.pp.moments(sim_data, n_pcs=30, n_neighbors=30)
    adatas.append(sim_data)

adata = anndata.concat(adatas)
adata.obs = adata.obs.reset_index()
    
    
    
    
    
