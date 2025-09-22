import os
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
        self.slice_data_loader, self.data_for_model_init = slice_data_loader
        self.token_mapping_model = slice_data_loader[0].gene_exp_model
        self.location_model = location_model
        self.subclass_model = subclass_model
        self.config = config

    # === LOCATION ===
    def infer_location(self, adata, new_tissue):
        loc_type = self.config["location_inference_type"]

        if loc_type == "model":
            # xyz = np.asarray(self.location_model.sample_slice_conditionally(
            #     new_tissue, size=len(new_tissue), interior=True
            # ))

            if self.config["data_mode"] == "rq2":
                xyz=self.location_model[0].sample_with_guidance(
                        self.slice_data_loader.train_slices[0].n_obs*3,
                        self.location_model[1],
                        conditional_z=new_tissue.obsm["aligned_spatial"].mean(0)[-1],
                        guidance_scale=0.0
                    )
                xyz = xyz[np.linalg.norm(xyz - self.slice_data_loader.hole_centers[0], axis=1) < 0.3][:len(new_tissue)]


            else:
                xyz=self.location_model[0].sample_with_guidance(
                        adata.n_obs,
                        self.location_model[1],
                        conditional_z=new_tissue.obsm["aligned_spatial"].mean(0)[-1],
                        guidance_scale=0.0
                    )
            print(xyz)
            mask = np.all(np.isfinite(xyz), axis=1)
            xyz = xyz[mask]


            plt.scatter(np.asarray(xyz)[:,0], np.asarray(xyz)[:,1], s=0.1, alpha=0.5)
            plt.xlim(0,12)
            plt.ylim(0,8)
            plt.title("Inferred Locations model")
            plt.savefig("artifacts/inferred_locations_model.png", dpi=300)

            plt.figure()

            # Ground truth
            xyz_gt = new_tissue.obsm["aligned_spatial"].copy()
            plt.scatter(np.asarray(xyz_gt)[:,0], np.asarray(xyz_gt)[:,1], s=0.1, alpha=0.5)
            plt.xlim(0,12)
            plt.ylim(0,8)
            plt.title("Ground Truth Locations")
            plt.savefig("artifacts/ground_truth_locations.png", dpi=300)

            plt.figure()

            closest_ref_slice = np.argmin([np.square(ref_slice.obsm["aligned_spatial"].mean(0)[-1] - new_tissue.obsm["aligned_spatial"].mean(0)[-1]) for ref_slice in self.slice_data_loader.reference_slices])
            cs = self.slice_data_loader.reference_slices[closest_ref_slice].obsm["aligned_spatial"].copy()
            cs[:, -1] = new_tissue.obsm["aligned_spatial"].mean(0)[-1]
            plt.scatter(np.asarray(cs)[:,0], np.asarray(cs)[:,1], s=0.1, alpha=0.5)
            plt.xlim(0,12)
            plt.ylim(0,8)
            plt.title("Closest Reference Slice")
            plt.savefig("artifacts/closest_reference_slice.png", dpi=300)
            plt.figure()




        elif loc_type == "closest_slice":
            closest_ref_slice = np.argmin([np.square(ref_slice.obsm["aligned_spatial"].mean(0)[-1] - new_tissue.obsm["aligned_spatial"].mean(0)[-1]) for ref_slice in self.slice_data_loader.reference_slices])
            xyz = self.slice_data_loader.reference_slices[closest_ref_slice].obsm["aligned_spatial"].copy()
            xyz[:, -1] = new_tissue.obsm["aligned_spatial"].mean(0)[-1]

        n = adata.n_obs
        if xyz.shape[0] > n:
            idx = np.random.choice(xyz.shape[0], size=n, replace=False)
            xyz = xyz[idx]
        elif xyz.shape[0] < n:
            idx = np.random.choice(xyz.shape[0], size=n, replace=True)
            xyz = xyz[idx]

        adata.obsm["spatial"] = np.asarray(xyz)

        return adata 

    # === CLUSTER / SUBCLASS ===
    def infer_cluster(self, adata, new_tissue):
        clust_type = self.config["cluster_inference_type"]

        xyz_samples = np.asarray(adata.obsm["spatial"])

        if "majority_baseline" in clust_type:
            ref_np = np.asarray(ad.concat(self.slice_data_loader.reference_slices).obsm["aligned_spatial"])
            labels_np = np.asarray(ad.concat(self.slice_data_loader.reference_slices).obs["token"]).astype(int)

            nbrs = NearestNeighbors(n_neighbors=20).fit(ref_np)
            _, indices = nbrs.kneighbors(xyz_samples)

            preds = np.zeros(len(xyz_samples), dtype=int)
            for i, neigh_idxs in enumerate(indices):
                counts = np.bincount(labels_np[neigh_idxs])
                preds[i] = np.argmax(counts)

            adata.obs["token"] = preds

        elif "model" in clust_type:
            xyz_samples_t = torch.tensor(xyz_samples, dtype=torch.float32, device="cuda")
            adata_sampled, preds, probs = generate_anndata_from_samples(
                self.subclass_model,
                xyz_samples_t,
                xyz_samples_t.device,
                sample_from_probs=True,
                use_conditionals=False,
                xyz_labels=None,
                num_classes=max(self.subclass_model.y) + 1,
                gibbs=False,
                n_iter=1,
                use_budget=False,
                graph_smooth=False,
            )
            adata.obs["token"] = adata_sampled.obs["token"].to_numpy()

        
        if "homogenize" in clust_type:
            homogenized_labels = homogenize(xyz_samples.cpu(), preds, k=100, n_iter=1,alpha=0.7,maximize=True, probs=probs)            
            adata.obs["token"] = homogenized_labels
        return adata


    def infer_expression(self, pred_data, real_data):
        import numpy as np
        import pandas as pd

        exp_type = self.config["expression_inference_type"]

        if exp_type == "averaging":
            rows = np.array(self.token_mapping_model.get_gene_exp_from_token(pred_data.obs["token"].tolist()))[:,0]
            obs = pred_data.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=real_data.var_names)

            # 4) re-create
            return AnnData(X=rows, obs=obs, var=var, obsm=pred_data.obsm.copy())

        elif exp_type == "lookup":
            # 1) build and subsample ref_tissue as before…
            ref_tissue = ad.concat(self.slice_data_loader.reference_slices)
            if getattr(self, "ref_sample", None):
                pct = self.ref_sample
                n   = int(len(ref_tissue) * pct / 100)
                ref_tissue = ref_tissue[np.random.choice(len(ref_tissue), size=n, replace=False)]

            # 2) extract arrays
            ref_ct  = np.array(ref_tissue.obs["token"].tolist())
            ref_pos = np.array(ref_tissue.obsm["aligned_spatial"])
            try:
                ref_exp = np.array(ref_tissue.X.todense())
            except:
                ref_exp = ref_tissue.X

            pred_ct  = np.array(pred_data.obs["token"].tolist())
            pred_pos = np.array(pred_data.obsm["spatial"])
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
                k   = min(1, len(idxs))
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
                obs=pred_data.obs.copy(),
                var = pd.DataFrame(index=real_data.var_names),       # ← use the original var with correct length
                obsm=pred_data.obsm.copy()
            )


        elif exp_type == "model":
            # pkl.dump(adata, open(f"artifacts/pred_expression_model_for_{self.config['data_mode']}_slice_{self.config['slice_index']}.pkl","wb"))
            # return pkl.load(open(f"artifacts/pred_expression_model_for_rq1_slice_0.pkl","rb"))

            # if os.path.exists(f"artifacts/pred_expression_model_for_{self.config['data_mode']}_slice_{self.config['slice_index']}.pkl"):
            #     pred_data = pkl.load(open(f"artifacts/pred_expression_model_for_{self.config['data_mode']}_slice_{self.config['slice_index']}.pkl","rb"))
            #     return pred_data


            from scMulan import model_generate
            import pandas as pd
            import scipy.sparse as sp
            import numpy as np
            import scanpy as sc
            import torch

            meta_info = torch.load(f'{self.slice_data_loader.metadata_dir}4hierarchy_metainfo_mouse.pt')

            # 1) preserve obs but add necessary technology metadata, etc.
            for name, default in zip(['organ', 'technology', 'species', 'disease_state'], ['brain','M550','mouse','healthy']):
                if name in real_data.obs.columns:
                    pred_data.obs[name] = real_data.obs[name]
                else:
                    pred_data.obs[name] = default
            # Get higher hierarchy levels

            ##IMP TO USE THIS ONE HERE
            pred_data.obs['readable_label'] = self.data_for_model_init.token_mapping_model.get_label_from_token(pred_data.obs['token'].values)
            hierarchy = pkl.load(open(f'{self.slice_data_loader.metadata_dir}hierarchy.pkl','rb'))
            for h in ['class', 'subclass','supertype','cluster']:
                pred_data.obs[h] = 'na'
            for cell in pred_data.obs_names:
                c, sc, st, cl = hierarchy[('cluster',pred_data.obs.loc[cell, 'readable_label'])]
                for h, v in zip(['class', 'subclass','supertype','cluster'], [c, sc, st, cl]):
                    pred_data.obs.loc[cell, h] = v
            # Get binned x,y,z
            coordfiles = [f'{self.slice_data_loader.metadata_dir}edges_x.pkl',
                          f'{self.slice_data_loader.metadata_dir}edges_y.pkl',
                          f'{self.slice_data_loader.metadata_dir}edges_z.pkl',
                         ]
            for coord, coordfile, i in zip(('x','y','z'),coordfiles, range(3)):
                vals_full = pred_data.obsm[f'spatial'][:,i].astype(float)
                edges = pkl.load(open(coordfile, 'rb'))
                # edges   = coord_bins[f'{coord}{coord_suffix}']
                bin_idxs = np.digitize(vals_full, edges, right=True)
                pred_data.obs[f'<{coord}>'] = bin_idxs
            obs = pred_data.obs.copy()
            

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=real_data.var_names)

            # 3) build brand-new X as zeros, sparse
            X_sparse = np.zeros((len(pred_data.obs_names),len(real_data.var_names)))

            # 4) re-create
            adata_sub = AnnData(X=X_sparse, obs=obs, var=var, obsm=pred_data.obsm.copy())
            ckp_path = '/compute/oven-0-13/skrieger/mouse-mediummodelscrna/epoch110_model.pt'
            scml = model_generate(ckp_path=ckp_path,
                                adata=adata_sub,
                                meta_info=meta_info,
                                use_kv_cache=True,
                                )
            results = scml.generate_cell_genesis(
                idx=range(len(pred_data.obs_names)),
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
            obs = pred_data.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=real_data.var_names)

            # 4) re-create
            adata = AnnData(X=rows, obs=obs, var=scml.adata.var, obsm=pred_data.obsm)
            print(adata)
            return adata


    # === ORCHESTRATOR ===
    def run_inference(self, adatas):
        real_data = ad.concat(adatas)

        # start with empty shell
        pred_data = AnnData(obs=pd.DataFrame(index=real_data.obs.index))

        # step 1: location
        if "end" in self.config["location_inference_type"]:
            return pred_data
        elif "skip" in self.config["location_inference_type"]:
            pred_data.obsm["spatial"] = real_data.obsm["aligned_spatial"].copy()
        else:
            pred_data = self.infer_location(pred_data, real_data)

        # step 2: cluster
        if "end" in self.config["cluster_inference_type"]:
            return pred_data
        elif "skip" in self.config["cluster_inference_type"]:
            pred_data.obs["token"] = real_data.obs["token"].copy()
        else:
            pred_data = self.infer_cluster(pred_data, real_data)

        if "end" in self.config["expression_inference_type"]:
            return pred_data
        elif "skip" in self.config["expression_inference_type"]:
            pred_data.X = real_data.X.copy()
        else:
            pred_data = self.infer_expression(pred_data, real_data)

        return pred_data
