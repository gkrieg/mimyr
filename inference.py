import copy
import math
import os
import pickle as pkl
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from anndata import AnnData
from matplotlib.pyplot import rc_context
from scipy.sparse import csr_matrix
from scipy.special import gamma
from scipy.spatial import cKDTree
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import *
from analysis import *

from generative_transformer.utils.hf_tokenizer import scMulanTokenizer
from generative_transformer.scMulan import (
    compute_global_bin_edges,
    generate_prompt_for_cg,
    scMulan,
    model_generate,
    model_inference,
)
import generative_transformer.scMulan as scMulan_module  # module handle if needed
from generative_transformer.model.model_kvcache import (
    MulanConfig,
    scMulanModel_kv as scMulanModel,
)

class Inferernce:
    def __init__(self, location_model, subclass_model, slice_data_loader, config):
        self.slice_data_loader = slice_data_loader
        # self.slice_data_loader2 = copy.deepcopy(slice_data_loader)
        # self.slice_data_loader2.gene_exp_model.id_to_subclass = pkl.load(open("id_to_subclass.pkl","rb"))
        self.token_mapping_model = slice_data_loader.gene_exp_model
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
                        guidance_scale=self.config["guidance_signal"]
                    )
                xyz = xyz[np.linalg.norm(xyz - self.slice_data_loader.hole_centers[0], axis=1) < 0.3][:len(new_tissue)]
            
            elif self.config["data_mode"] == "rq3":
                xyz=self.location_model[0].sample_with_guidance(
                        adata.n_obs*2,
                        self.location_model[1],
                        conditional_z=new_tissue.obsm["aligned_spatial"].mean(0)[-1],
                        guidance_scale=self.config["guidance_signal"]
                    )
                xyz=xyz[xyz[:,0]<new_tissue.obsm["aligned_spatial"].max(0)[0]]

            elif self.config["data_mode"] == "rq4":
                xyz=self.location_model[0].sample_with_guidance(
                        adata.n_obs,
                        self.location_model[1],
                        conditional_x=new_tissue.obsm["aligned_spatial"].mean(0)[0],
                        guidance_scale=self.config["guidance_signal"]
                    )



            else:
                xyz=self.location_model[0].sample_with_guidance(
                        adata.n_obs,
                        self.location_model[1],
                        conditional_z=new_tissue.obsm["aligned_spatial"].mean(0)[-1],
                        guidance_scale=self.config["guidance_signal"]
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

        elif loc_type == "uniform_circle":

            def sample_circle(radius=0.3, n=1):
                theta = np.random.uniform(0, 2*np.pi, n)
                r = radius * np.sqrt(np.random.uniform(0, 1, n))
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                return np.column_stack((x, y))

            # Example: 5 samples
            xyz = sample_circle(0.3, adata.n_obs)
            # make the third column the same as the mean z of the new tissue
            xyz = np.column_stack((xyz, np.full(adata.n_obs, new_tissue.obsm["aligned_spatial"].mean(0)[-1])))            
            xyz+=self.slice_data_loader.hole_centers[0]
            


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
                num_classes= 5274, #max(self.subclass_model.y) + 1,
                gibbs=False,
                n_iter=1,
                use_budget=False,
                graph_smooth=False,
            )
            adata.obs["token"] = adata_sampled.obs["token"].to_numpy()
            # Percentage of cells not in training data 


        
        if "homogenize" in clust_type:
            homogenized_labels = homogenize(xyz_samples.cpu(), preds, k=100, n_iter=1,alpha=0.7,maximize=True, probs=probs)            
            adata.obs["token"] = homogenized_labels
        return adata


    def infer_expression(self, pred_data, real_data):

        exp_type = self.config["expression_inference_type"]

        if exp_type == "averaging":
            rows = np.array(self.token_mapping_model.get_gene_exp_from_token(pred_data.obs["token"].tolist()))[:,0]
            obs = pred_data.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=real_data.var_names)

            # 4) re-create
            return AnnData(X=rows, obs=obs, var=var, obsm=pred_data.obsm.copy()), None

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
            ), None


        elif exp_type == "model":
            # pkl.dump(adata, open(f"artifacts/pred_expression_model_for_{self.config['data_mode']}_slice_{self.config['slice_index']}.pkl","wb"))
            # return pkl.load(open(f"artifacts/pred_expression_model_for_rq1_slice_0.pkl","rb"))

            # if os.path.exists(f"artifacts/pred_expression_model_for_{self.config['data_mode']}_slice_{self.config['slice_index']}.pkl"):
            #     pred_data = pkl.load(open(f"artifacts/pred_expression_model_for_{self.config['data_mode']}_slice_{self.config['slice_index']}.pkl","rb"))
            #     return pred_data


            
            meta_info_path = f'{self.slice_data_loader.metadata_dir}{self.config["meta_info"]}'
            meta_info = torch.load(meta_info_path)

            # 1) preserve obs but add necessary technology metadata, etc.
            for name, default in zip(['organ', 'technology', 'species', 'disease_state'], ['brain','M550','mouse','healthy']):
                if name in real_data.obs.columns:
                    pred_data.obs[name] = real_data.obs[name]
                else:
                    pred_data.obs[name] = default
            # Get higher hierarchy levels

            ##IMP TO USE THIS ONE HERE
            pred_data.obs['readable_label'] = self.slice_data_loader.gene_exp_model.get_label_from_token(pred_data.obs['token'].values)
            hierarchy = pkl.load(open(f'{self.slice_data_loader.metadata_dir}hierarchy.pkl','rb'))
            for h in ['class', 'subclass','supertype','cluster']:
                pred_data.obs[h] = 'na'
            for cell in pred_data.obs_names:
                try:
                    c, sc, st, cl = hierarchy[('cluster',pred_data.obs.loc[cell, 'readable_label'])]
                except:
                    c, sc, st, cl = ('01 IT-ET Glut',	'006 L4/5 IT CTX Glut',	'0027 L4/5 IT CTX Glut_5',	'0097 L4/5 IT CTX Glut_5')
                    print('skip', pred_data.obs.loc[cell, 'readable_label'])
                for h, v in zip(['class', 'subclass','supertype','cluster'], [c, sc, st, cl]):
                    pred_data.obs.loc[cell, h] = v
            # Get binned x,y,z
            # coordfiles = [f'{self.slice_data_loader.metadata_dir}edges_x.pkl',
            #               f'{self.slice_data_loader.metadata_dir}edges_y.pkl',
            #               f'{self.slice_data_loader.metadata_dir}edges_z.pkl',
            #              ]
            # for coord, coordfile, i in zip(('x','y','z'),coordfiles, range(3)):
            #     vals_full = pred_data.obsm[f'spatial'][:,i].astype(float)
            #     edges = pkl.load(open(coordfile, 'rb'))
            #     # edges   = coord_bins[f'{coord}{coord_suffix}']
            #     bin_idxs = np.digitize(vals_full, edges, right=True)
            #     pred_data.obs[f'<{coord}>'] = bin_idxs
            # pred_data.obs['<x>'] = real_data.obs.loc[pred_data.obs_names,'<x>']
            # pred_data.obs['<y>'] = real_data.obs.loc[pred_data.obs_names,'<y>']
            # pred_data.obs['<z>'] = real_data.obs.loc[pred_data.obs_names,'<z>']

            ## USE PRED DATA HERE
            pred_data.obs['x'] = pred_data.obsm['spatial'][:,2]
            pred_data.obs['y'] = pred_data.obsm['spatial'][:,1]
            pred_data.obs['z'] = pred_data.obsm['spatial'][:,0]

            coordfiles = [f'{self.slice_data_loader.metadata_dir}edges_x.pkl',
                          f'{self.slice_data_loader.metadata_dir}edges_y.pkl',
                          f'{self.slice_data_loader.metadata_dir}edges_z.pkl',
                         ]
            for coord, coordfile in zip(('x','y','z'),coordfiles):
                vals_full = pred_data.obs[f'{coord}']
                edges = pkl.load(open(coordfile, 'rb'))
                bin_idxs = np.digitize(vals_full, edges, right=True)
                pred_data.obs[f'<{coord}>'] = bin_idxs



            obs = pred_data.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=real_data.var_names)

            # 3) build brand-new X as zeros, sparse
            X_sparse = np.zeros((len(pred_data.obs_names),len(real_data.var_names)))

            # 4) re-create
            adata_sub = AnnData(X=X_sparse, obs=obs, var=var, obsm=pred_data.obsm.copy())
            ckp_path = self.config["expression_model_checkpoint"]#'/compute/oven-0-13/skrieger/mouse-mediummodelscrna/epoch110_model.pt'
            scml = model_inference(ckp_path=ckp_path,
                                adata=adata_sub,
                                meta_info=meta_info,
                                use_kv_cache=True,
                                )
            results = scml.generate_cell_genesis(
                idx=range(len(pred_data.obs_names)),
                max_new_tokens=500,      #### SET TO 500 FOR GENE UNION
                top_k=5,
                verbose=False,
                return_gt=False,
                batch_size=4096,#128,    ## CHECK THIS SUSPICIOUSLY HIGH BATCH SIZE
                cheat_with_tokens=None,
                cheat_with_expr=None,
            )
            rows = [r[0] for r in results]

            rows = np.array(rows)
            obs = pred_data.obs.copy()


            # 4) re-create
            adata = AnnData(X=rows, obs=obs, var=scml.adata.var, obsm=pred_data.obsm)
            return adata, results


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

        plt.figure()

        plt.scatter(pred_data.obsm["spatial"][:,(2 if self.config["data_mode"] == "rq4" else 0)], pred_data.obsm["spatial"][:,1], s=0.1, alpha=0.5)
        plt.xlim(0,12)
        plt.ylim(0,8)
        plt.title("Inferred Locations model")
        plt.savefig(f"{self.config['artifact_dir']}/inferred_locations_model.png", dpi=300)

        plt.figure()

        # Ground truth
        plt.scatter(real_data.obsm["aligned_spatial"][:,(2 if self.config["data_mode"] == "rq4" else 0)], real_data.obsm["aligned_spatial"][:,1], s=0.1, alpha=0.5)
        plt.xlim(0,12)
        plt.ylim(0,8)
        plt.title("Ground Truth Locations")
        plt.savefig(f"{self.config['artifact_dir']}/ground_truth_locations.png", dpi=300)

        plt.figure()



        # step 2: cluster
        if "end" in self.config["cluster_inference_type"]:
            return pred_data
        elif "skip" in self.config["cluster_inference_type"]:
            pred_data.obs["token"] = real_data.obs["token"].copy()
        else:
            pred_data = self.infer_cluster(pred_data, real_data)

        
        # step 3: expression
        real_data.obsm["spatial"] = real_data.obsm["aligned_spatial"]
        assign_shared_colors([real_data,pred_data], color_key="token")
        plot_spatial_with_palette(real_data, color_key="token", spot_size=0.003, figsize=(10,10),save=f"./{self.config['artifact_dir']}/real_data_clusters.png", saggital="rq4" in self.config["data_mode"])
        plot_spatial_with_palette(pred_data, color_key="token", spot_size=0.003, figsize=(10,10),save=f"./{self.config['artifact_dir']}/pred_data_clusters.png", saggital="rq4" in self.config["data_mode"])



        if "end" in self.config["expression_inference_type"]:
            return pred_data
        elif "skip" in self.config["expression_inference_type"]:
            pred_data.X = real_data.X.copy()
        else:
            pred_data, res = self.infer_expression(pred_data, real_data)

        # do pca of X and create RGD colors for each cell and the plot pred and GT


        
        def joint_pca_rgb(real_data, pred_data, n_components=3):
            common_genes_atleast1 = ['Prkcq', 'Syt6', 'Ptprm', 'Hspg2', 'Cxcl14', 'Dock5', 'Stxbp6', 'Nfib', 'Gfap', 'Gja1', 'Tcf7l2', 'Rorb', 'Aqp4', 'Slc7a10', 'Grm3', 'Slc1a3', 'Serpine2', 'Lgr6', 'Slc32a1', 'Adamts19', 'Cdh20', 'Sox2', 'Lpar1', 'Pcp4l1', 'Spock3', 'Lypd1', 'Zeb2', 'Unc13c', 'Rgs6', 'Sox6', 'Tafa2', 'Lrp4', 'St6galnac5', 'C030029H02Rik', 'Ust', '2900052N01Rik', 'Sp8', 'Igf2', 'Fli1', 'Opalin', 'Sox10', 'Acta2', 'Chrm2', 'Gad2', 'Cgnl1', 'Vcan', 'Cldn5', 'Mog', 'Maf', 'Bmp4', 'Ctss', 'Dach1', 'Grm8', 'Zfp536', 'Zic1', 'Bcl11b', 'Prkd1', 'C1ql1', 'Hs3st4', 'Pdgfd', 'Nxph1', 'Ebf1', 'Klk6', 'Man1a', 'Sema3c', 'Nr2f2', 'Tgfbr2', 'Pde3a', 'Zfpm2', 'C1ql3', 'Marcksl1', 'Gli2', 'Sema5a', 'Wls', 'Hmcn1', 'Abcc9', 'Kcnip1', 'Mecom', 'Tshz2', 'Nfix', 'Gli3', 'Meis1', 'Kcnmb2', 'Egfem1', 'Adamtsl1', 'Tbx3', 'Gfra1', 'Fign', 'Glis3', 'Kcnj8', 'Adgrf5', 'Vip', 'Chn2', 'Tafa1', 'Ntng1', 'Grik1', 'St18', 'Rmst', 'Dscaml1', 'Synpr', 'Adra1a', 'Prom1', 'Cpa6']
            common_genes = pd.Index(common_genes_atleast1, name="gene")
            X_real = real_data[:,common_genes].X.toarray() if hasattr(real_data.X, "toarray") else real_data[:,common_genes].X
            X_pred = pred_data[:,common_genes].X.toarray() if hasattr(pred_data.X, "toarray") else pred_data[:,common_genes].X

            

            X_concat = np.vstack([X_real, X_pred])
            pca = PCA(n_components=n_components)
            comps = pca.fit_transform(X_concat)

            scaler = MinMaxScaler()
            comps_scaled = scaler.fit_transform(comps)

            rgb_real = comps_scaled[:len(real_data), :3]
            rgb_pred = comps_scaled[len(real_data):, :3]

            return rgb_real, rgb_pred

        # apply
        rgb_real, rgb_pred = joint_pca_rgb(real_data, pred_data)
        rgb_real = np.clip(rgb_real, 0, 1)
        rgb_pred = np.clip(rgb_pred, 0, 1)

        real_data.obs["rgb"] = [tuple(c) for c in rgb_real]
        pred_data.obs["rgb"] = [tuple(c) for c in rgb_pred]





        # plot real
        plt.figure(figsize=(10,10))
        plt.scatter(real_data.obsm["aligned_spatial"][:,(2 if self.config["data_mode"] == "rq4" else 0)],
                    real_data.obsm["aligned_spatial"][:,1],
                    c=rgb_real, s=0.1)
        plt.title("Real Data PCA Colors")
        plt.axis("equal")
        plt.savefig(f"{self.config['artifact_dir']}/real_data_pca.png", dpi=300)

        # plot pred
        plt.figure(figsize=(10,10))
        plt.scatter(pred_data.obsm["spatial"][:,(2 if self.config["data_mode"] == "rq4" else 0)],
                    pred_data.obsm["spatial"][:,1],
                    c=rgb_pred, s=400)
        plt.title("Pred Data PCA Colors")
        plt.axis("equal")
        plt.savefig(f"{self.config['artifact_dir']}/pred_data_pca.png", dpi=300)


        return pred_data
