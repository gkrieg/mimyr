import torch
from torch.utils.data import DataLoader
import anndata as ad
from matplotlib.pyplot import rc_context
from metrics import *
from analysis import *
import sys

sys.path.append("generative_transformer")
import tqdm
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import anndata as ad
import numpy as np
from scipy.spatial import cKDTree
import tqdm
import math
from sklearn.neighbors import KernelDensity
from scipy.special import gamma
import scanpy as sc
from scMulan import generate_prompt_for_cg

from utils.hf_tokenizer import scMulanTokenizer

from generative_transformer.scMulan import compute_global_bin_edges
import scMulan
import torch
from model.model_kvcache import MulanConfig, scMulanModel_kv as scMulanModel
from scMulan import scMulan, model_generate

from anndata import AnnData
from scipy.sparse import csr_matrix
import pandas as pd
from reference.GeneSymbolUniform.pyGSUni import GeneSymbolUniform

import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from anndata import AnnData
import tqdm

from sklearn.neighbors import NearestNeighbors


class Inferernce:
    def __init__(self, location_model, subclass_model, slice_data_loader, config):
        self.slice_data_loader = slice_data_loader
        self.token_mapping_model = slice_data_loader.gene_exp_model
        self.location_model = location_model
        self.subclass_model = subclass_model
        self.do_infer_location = config["infer_location"]
        self.location_inference_type = config["location_inference_type"]
        self.do_infer_subclass = config["infer_subclass"]
        self.subclass_inference_type = config["subclass_inference_type"]
        self.homogenize_subclass = config["homogenize_subclass"]

        self.do_infer_gene_expression = config["infer_gene_expression"]
        self.expression_inference_type = config["expression_inference_type"]

        # self.do_infer_subclass = self.do_infer_subclass or self.do_infer_location
        # self.do_infer_gene_expression = (
        #     self.do_infer_gene_expression
        #     or self.do_infer_location
        #     or self.do_infer_subclass
        # )

    def infer_location(self, new_tissue):
        if self.location_inference_type == "model":
            return self.location_model.sample_slice_conditionally(
                new_tissue, size=len(new_tissue), interior=True
            )

    def infer_subclass(self, xyz_samples):
        if self.subclass_inference_type == "majority_baseline":
            def to_numpy(x):
                if isinstance(x, np.ndarray):
                    return x
                try:
                    return x.cpu().numpy()
                except:
                    return np.asarray(x)

            test_np = to_numpy(xyz_samples)
            ref_np = to_numpy(ad.concat(self.slice_data_loader.reference_slices).obsm["aligned_spatial"])
            labels_np = to_numpy(ad.concat(self.slice_data_loader.reference_slices).obs["token"]).astype(int)

            # Build kNN index on reference locations
            nbrs = NearestNeighbors(n_neighbors=20).fit(ref_np)
            distances, indices = nbrs.kneighbors(test_np)  # indices shape = (N_test, k)

            n_test = test_np.shape[0]
            preds = np.zeros(n_test, dtype=int)
            probs = []
            for i in range(n_test):
                neighbor_idxs = indices[i]                          # shape (k,)
                neighbor_labels = labels_np[neighbor_idxs]          # shape (k,)
                # Compute majority vote (mode)
                counts = np.bincount(neighbor_labels)
                preds
                
                preds[i] = np.argmax(counts)

            adata_sampled = ad.AnnData(X=np.zeros((xyz_samples.shape[0], 1)))
            adata_sampled.obsm["spatial"] = xyz_samples.cpu().numpy()
            adata_sampled.obs["token"] = preds

        if self.subclass_inference_type == "model":
            adata_sampled, preds, probs = generate_anndata_from_samples(
                self.subclass_model,
                xyz_samples,
                xyz_samples.device,
                sample_from_probs=True,
                use_conditionals=False,
                xyz_labels=None,#new_tissue.obs["token"],
                num_classes=max(self.subclass_model.y)+1,
                gibbs=False,
                n_iter=1,
                use_budget=False,
                graph_smooth=False,
            )
        if self.homogenize_subclass:
            homogenized_labels = homogenize(xyz_samples.cpu(), preds, k=100, n_iter=1,alpha=0.7,maximize=True, probs=probs)            
            adata_sampled.obs["token"] = homogenized_labels
        for k in ['class', 'subclass', 'supertype', 'cluster']:
            if k in adata_sampled.obs.columns:
                del adata_sampled.obs[k]
        return adata_sampled

    def infer_expression(self, adatas, adata_sampled):
        import numpy as np
        import pandas as pd

        if self.expression_inference_type == "averaging":
            rows = np.array(self.token_mapping_model.get_gene_exp_from_token(adata_sampled.obs["token"].tolist()))[:,0]
            obs = adata_sampled.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=adatas[0].var_names)

            # 4) re-create
            return AnnData(X=rows, obs=obs, var=var, obsm=adata_sampled.obsm.copy())

        elif self.expression_inference_type == "lookup":
            # 1) build and subsample ref_tissue as before…
            ref_tissue = ad.concat(self.slice_data_loader.reference_slices)
            if getattr(self, "ref_sample", None):
                pct = self.ref_sample
                n   = int(len(ref_tissue) * pct / 100)
                ref_tissue = ref_tissue[np.random.choice(len(ref_tissue), size=n, replace=False)]

            # 2) extract arrays
            ref_ct  = np.array(ref_tissue.obs["token"].tolist())
            ref_pos = np.array(ref_tissue.obsm["aligned_spatial"])
            ref_exp = np.array(ref_tissue.X.todense())

            pred_ct  = np.array(adata_sampled.obs["token"].tolist())
            pred_pos = np.array(adata_sampled.obsm["spatial"])
            n_pred   = len(pred_pos)
            n_genes  = ref_exp.shape[1]

            # 3) index reference cells by type
            from collections import defaultdict
            ref_by_type = defaultdict(list)
            for i, ct in enumerate(ref_ct):
                ref_by_type[ct].append(i)

            # 4) build one KDTree per type
            trees = {}
            for ct, idxs in ref_by_type.items():
                trees[ct] = (cKDTree(ref_pos[idxs]), idxs)

            # 5) batch‑query per type
            pred_lookup = np.zeros((n_pred, n_genes), dtype=ref_exp.dtype)
            for ct, (tree, idxs) in trees.items():
                # find all pred cells of this type
                mask = (pred_ct == ct)
                if not mask.any():
                    continue
                pts = pred_pos[mask]
                k   = min(20, len(idxs))
                dists, nbrs = tree.query(pts, k=k)
                # ensure shape (N, k)
                if k == 1:
                    nbrs = nbrs[:, None]
                # gather and average
                selected = ref_exp[idxs][nbrs]     # shape (N, k, n_genes)
                means    = selected.mean(axis=1)   # shape (N, n_genes)
                pred_lookup[mask] = means

            # 6) wrap up
            return AnnData(
                X=pred_lookup,
                obs=adata_sampled.obs.copy(),
                var = pd.DataFrame(index=adatas[0].var_names),       # ← use the original var with correct length
                obsm=adata_sampled.obsm.copy()
            )


        elif self.expression_inference_type == "model":
            from scMulan import model_generate
            import pandas as pd
            import scipy.sparse as sp
            import numpy as np
            import scanpy as sc
            import torch
            def harmonize_dataset(adata, meta_info, organ='Brain', technology='M550', coord_suffix='_ccf', n_bins=100):
                if adata.X.max() > 10:
                    sc.pp.normalize_total(adata, target_sum=1e4) 
                    sc.pp.log1p(adata)
                sc.pp.filter_cells(adata,min_genes=10)
                coord_bins = {}
                for i,coord in enumerate(('x','y','z')):
                    vals_full = adata.obsm["spatial"][:,i]
                    vals = adata.obsm["spatial"][:,i]
                    coord_bins[f'{coord}{coord_suffix}'] = np.linspace(vals.min(), vals.max(), n_bins)
                    edges   = coord_bins[f'{coord}{coord_suffix}']
                    bin_idxs = np.digitize(vals_full, edges, right=True)
                    adata.obs[f'<{coord}>'] = bin_idxs
                adata.obs['organ'] = organ
                adata.obs['technology'] = technology
                cols = ['<x>','<y>','<z>','organ', 'class', 'subclass','supertype','cluster', 'technology']
                mask = adata.obs[cols].notnull().all(axis=1)
                adata = adata[mask].copy()

                existing_genes = adata.var_names.tolist()

                gene_set = list(meta_info['gene_set'])
                new_genes = [g for g in gene_set if g not in existing_genes]
                
                print(f"Adding {len(new_genes)} new genes to adata")
                
                if len(new_genes) > 0:
                    n_obs = adata.n_obs
                    n_new_vars = len(new_genes)
                
                    new_vars = pd.DataFrame(index=new_genes)
                    new_var = pd.concat([adata.var, new_vars], axis=0)
                    
                
                    if sp.issparse(adata.X):
                        new_data = sp.csr_matrix((n_obs, n_new_vars))
                        X = sp.hstack([adata.X, new_data], format='csr')
                    else:
                        new_data = np.zeros((n_obs, n_new_vars))
                        X = np.hstack([adata.X, new_data])
                
                    new_adata = sc.AnnData(X=X, var=new_var, obs=adata.obs, obsm=adata.obsm, uns=adata.uns)
                    new_adata.var_names_make_unique()
                    del adata
                    adata = new_adata
                return adata

            meta_info = torch.load('/work/magroup/skrieger/tissue_generator/MERFISH_aging/4hierarchy_metainfo_mouse.pt')

            # 1) preserve obs
            adata_sub = adata_sampled
            obs = adata_sub.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=adatas[0].var_names)

            # 3) build brand-new X as zeros, sparse
            X_sparse = adatas[0].X

            # 4) re-create
            adata_sub = AnnData(X=X_sparse, obs=obs, var=var, obsm=adata_sampled.obsm.copy())

            harmonized_adata = harmonize_dataset(adata_sub,meta_info)

            ckp_path = '/compute/oven-0-13/skrieger/scMulan-output-mouse-ST-small/epoch43_model.pt'
            scml = model_generate(ckp_path=ckp_path,
                                adata=harmonized_adata,
                                meta_info=meta_info,
                                use_kv_cache=True,
                                )
            rows = []
            results = scml.generate_cell_genesis(
                        idx=[10,12],
                        max_new_tokens= 200,
                        top_k=5,
                        verbose=False,
                        return_gt=True,
                        batch_size=24,
                    )
            rows = [r[0] for r in results]

            rows = np.array(rows)
            obs = adata_sampled.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=adatas[0].var_names)

            # 4) re-create
            return AnnData(X=rows, obs=obs, var=var)

    def run_inference(self, adatas):
        new_tissue = ad.concat(adatas)
        device = "cuda"

        if self.do_infer_location:
            xyz_samples = torch.tensor(
                self.infer_location(new_tissue), dtype=torch.float32
            ).to(device)
        else:
            xyz_samples = torch.tensor(
                new_tissue.obsm["aligned_spatial"], dtype=torch.float32
            ).to(device)

        if self.do_infer_subclass:
            adata_sampled = self.infer_subclass(xyz_samples)
        else:
            adata_sampled = new_tissue

        if self.do_infer_gene_expression:
            adata_sampled = self.infer_expression(adatas, adata_sampled)

        return adata_sampled