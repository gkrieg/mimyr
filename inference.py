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
import pickle as pkl

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
            print(adatas)
            print(adata_sampled)
            rows = np.array(self.token_mapping_model.get_gene_exp_from_token(adata_sampled.obs["token"].tolist()))[:,0]
            print(rows.shape)
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

            meta_info = torch.load(f'{self.slice_data_loader.metadata_dir}4hierarchy_metainfo_mouse.pt')

            # 1) preserve obs but add necessary technology metadata, etc.
            for name, default in zip(['organ', 'technology', 'species', 'disease_state'], ['brain','M550','mouse','healthy']):
                if name in adatas[0].obs.columns:
                    adata_sampled.obs[name] = adatas[0].obs[name]
                else:
                    adata_sampled.obs[name] = default
            # Get higher hierarchy levels
            adata_sampled.obs['readable_label'] = self.token_mapping_model.get_label_from_token(adata_sampled.obs['token'].values)
            hierarchy = pkl.load(open(f'{self.slice_data_loader.metadata_dir}hierarchy.pkl','rb'))
            for h in ['class', 'subclass','supertype','cluster']:
                adata_sampled.obs[h] = 'na'
            for cell in adata_sampled.obs_names:
                c, sc, st, cl = hierarchy[('cluster',adata_sampled.obs.loc[cell, 'readable_label'])]
                for h, v in zip(['class', 'subclass','supertype','cluster'], [c, sc, st, cl]):
                    adata_sampled.obs.loc[cell, h] = v
            # Get binned x,y,z
            coordfiles = [f'{self.slice_data_loader.metadata_dir}edges_x.pkl',
                          f'{self.slice_data_loader.metadata_dir}edges_y.pkl',
                          f'{self.slice_data_loader.metadata_dir}edges_z.pkl',
                         ]
            for coord, coordfile, i in zip(('x','y','z'),coordfiles, range(3)):
                vals_full = adata_sampled.obsm[f'spatial'][:,i].astype(float)
                edges = pkl.load(open(coordfile, 'rb'))
                # edges   = coord_bins[f'{coord}{coord_suffix}']
                bin_idxs = np.digitize(vals_full, edges, right=True)
                adata_sampled.obs[f'<{coord}>'] = bin_idxs
            obs = adata_sampled.obs.copy()
            

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=adatas[0].var_names)

            # 3) build brand-new X as zeros, sparse
            X_sparse = np.zeros((len(adata_sampled.obs_names),len(adatas[0].var_names)))

            # 4) re-create
            adata_sub = AnnData(X=X_sparse, obs=obs, var=var, obsm=adata_sampled.obsm.copy())
            ckp_path = '/compute/oven-0-13/skrieger/mouse-mediummodelscrna/epoch110_model.pt'
            scml = model_generate(ckp_path=ckp_path,
                                adata=adata_sub,
                                meta_info=meta_info,
                                use_kv_cache=True,
                                )
            results = scml.generate_cell_genesis(
                idx=range(len(adata_sampled.obs_names)),
                max_new_tokens=200,
                top_k=5,
                verbose=False,
                return_gt=False,
                batch_size=128,
                cheat_with_tokens=None,
                cheat_with_expr=None,
            )
            rows = [r[0] for r in results]

            rows = np.array(rows)
            obs = adata_sampled.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=adatas[0].var_names)

            # 4) re-create
            adata = AnnData(X=rows, obs=obs, var=scml.adata.var, obsm=adata_sampled.obsm)
            print(adata)
            return adata

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
            adata_sampled.obsm['spatial'] = new_tissue.obs.loc[:,['z_ccf','y_ccf','x_ccf']].values
        if self.do_infer_gene_expression:
            adata_sampled = self.infer_expression(adatas, adata_sampled)

        return adata_sampled