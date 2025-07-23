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
                num_classes=self.subclass_model.x.shape[1] - 3,
                gibbs=False,
                n_iter=1,
                use_budget=False,
                graph_smooth=False,
            )
        if self.homogenize_subclass:
            homogenized_labels = homogenize(xyz_samples.cpu(), preds, k=100, n_iter=1,alpha=0.7,maximize=True, probs=probs)            
            adata_sampled.obs["token"] = homogenized_labels
        return adata_sampled

    def infer_expression(self, adatas, adata_sampled):
        if self.expression_inference_type == "averaging":
            rows = np.array(self.token_mapping_model.get_gene_exp_from_token(adata_sampled.obs["token"].tolist()))[:,0]
            obs = adata_sampled.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=adatas[0].var_names)

            # 4) re-create
            return AnnData(X=rows, obs=obs, var=var, obsm=adata_sampled.obsm.copy())

        elif self.expression_inference_type == "lookup":

            # 1) build and optionally subsample reference tissue
            ref_tissue = ad.concat(self.slice_data_loader.reference_slices)
            if getattr(self, "ref_sample", None):
                pct = self.ref_sample
                n   = int(len(ref_tissue) * pct / 100)
                idx = np.random.choice(len(ref_tissue), size=n, replace=False)
                ref_tissue = ref_tissue[idx]

            # 2) extract GT, ref and pred data
            ref_ct  = ref_tissue.obs["token"].tolist()
            ref_pos = np.array(ref_tissue.obsm["aligned_spatial"])
            ref_exp = np.array(ref_tissue.X.todense())

            pred_ct   = adata_sampled.obs["token"].tolist()
            pred_pos  = np.array(adata_sampled.obsm["spatial"])
            n_pred    = len(pred_pos)
            n_genes   = ref_exp.shape[1]
            pred_lookup = np.zeros((n_pred, n_genes), dtype=ref_exp.dtype)

            # 3) index reference cells by type
            ref_by_type = {}
            for i, ct in enumerate(ref_ct):
                ref_by_type.setdefault(ct, []).append(i)

            # 4) lookup average expression
            for i, (ct, pos) in tqdm.tqdm(enumerate(zip(pred_ct, pred_pos)), total=n_pred):
                if ct not in ref_by_type:
                    continue
                idxs    = ref_by_type[ct]
                subset  = ref_pos[idxs]
                tree    = cKDTree(subset)
                k_query = min(20, len(subset))
                _, nbrs = tree.query(pos, k=k_query)
                if k_query == 1:
                    nbrs = [nbrs]
                selected = ref_exp[idxs][nbrs]
                pred_lookup[i] = selected.mean(axis=0)

            # 5) package into AnnData and return
            rows = pred_lookup
            obs  = adata_sampled.obs.copy()
            var  = pd.DataFrame(index=adata_sampled.var_names)
            var=pd.DataFrame(index=adatas[0].var_names)
            return AnnData(X=rows, obs=obs, var=var, obsm=adata_sampled.obsm.copy())

        elif self.expression_inference_type == "model":
            # 1) preserve obs
            adata_sub = adata_sampled
            obs = adata_sub.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=adatas[0].var_names)

            # 3) build brand-new X as zeros, sparse
            X_sparse = adatas[0].X

            # 4) re-create
            adata_sub = AnnData(X=X_sparse, obs=obs, var=var)

            coord_bins = {}
            n_bins = 10  # e.g. 10

            adata_sub.obsm["spatial"] = adata_sampled.obsm["aligned_spatial"]
            adata_sub.obs["x"] = adata_sub.obsm["spatial"][:, 0]
            adata_sub.obs["y"] = adata_sub.obsm["spatial"][:, 1]
            adata_sub.obs["z"] = adata_sub.obsm["spatial"][:, 2]

            adata_sub.obs["organ"] = "Brain"

            region_map = {
                "SS-GU-VISC": "Cerebral cortex",  # somatosensory gustatory/visceral area
                "PL-ILA-ORB": "Cerebral cortex",  # prelimbic, infralimbic, orbital prefrontal
                "TEa-PERI-ECT": "Cerebral cortex",  # temporal association & perirhinal
                "MOp": "Cerebral cortex",  # primary motor cortex
                "VIS": "Cerebral cortex",  # visual cortex
                "VIS-PTLp": "Cerebral cortex",  # posterior lateral visual
                "SSp": "Cerebral cortex",  # primary somatosensory cortex
                "MO-FRP": "Cerebral cortex",  # frontal pole motor
                "AI": "Cerebral cortex",  # agranular insular cortex
                "AUD": "Cerebral cortex",  # auditory cortex
                "ACA": "Cingulate cortex",  # anterior cingulate area
                "RSP": "Cingulate cortex",  # retrosplenial cortex
                np.nan: "Unclassified",  # missing values
            }
            # adata_sub.obs['region'] = adata_sub.obs['region_of_interest_acronym'].map(region_map)
            # adata_sub.obs['region'] = adata_sub.obs['region'].fillna('Unclassified')
            new2existing = {
                # glutamatergic → Excitatory neuron
                "006 L4/5 IT CTX Glut": "Excitatory neuron",
                "030 L6 CT CTX Glut": "Excitatory neuron",
                "029 L6b CTX Glut": "Excitatory neuron",
                "032 L5 NP CTX Glut": "Excitatory neuron",
                "005 L5 IT CTX Glut": "Excitatory neuron",
                "007 L2/3 IT CTX Glut": "Excitatory neuron",
                "022 L5 ET CTX Glut": "Excitatory neuron",
                "021 L4 RSP-ACA Glut": "Excitatory neuron",
                "004 L6 IT CTX Glut": "Excitatory neuron",
                "020 L2/3 IT RSP Glut": "Excitatory neuron",
                "001 CLA-EPd-CTX Car3 Glut": "Excitatory neuron",
                "003 L5/6 IT TPE-ENT Glut": "Excitatory neuron",
                "028 L6b/CT ENT Glut": "Excitatory neuron",
                "002 IT EP-CLA Glut": "Excitatory neuron",
                "025 CA2-FC-IG Glut": "Excitatory neuron",
                "009 L2/3 IT PIR-ENTl Glut": "Excitatory neuron",
                "010 IT AON-TT-DP Glut": "Excitatory neuron",
                "036 HPF CR Glut": "Excitatory neuron",
                "008 L2/3 IT ENT Glut": "Excitatory neuron",
                "027 L6b EPd Glut": "Excitatory neuron",
                "114 COAa-PAA-MEA Barhl2 Glut": "Excitatory neuron",
                "018 L2 IT PPP-APr Glut": "Excitatory neuron",
                "035 OB Eomes Ms4a15 Glut": "Excitatory neuron",
                "262 Pineal Crx Glut": "Excitatory neuron",
                "034 NP PPP Glut": "Excitatory neuron",
                "019 L2/3 IT PPP Glut": "Excitatory neuron",
                "033 NP SUB Glut": "Excitatory neuron",
                "115 MS-SF Bsx Glut": "Excitatory neuron",
                "016 CA1-ProS Glut": "Excitatory neuron",
                # GABAergic → Inhibitory neuron
                "053 Sst Gaba": "Inhibitory neuron",
                "050 Lamp5 Lhx6 Gaba": "Inhibitory neuron",
                "046 Vip Gaba": "Inhibitory neuron",
                "052 Pvalb Gaba": "Inhibitory neuron",
                "049 Lamp5 Gaba": "Inhibitory neuron",
                "047 Sncg Gaba": "Inhibitory neuron",
                "056 Sst Chodl Gaba": "Inhibitory neuron",
                "041 OB-in Frmd7 Gaba": "Inhibitory neuron",
                "066 NDB-SI-ant Prdm12 Gaba": "Inhibitory neuron",
                "061 STR D1 Gaba": "Inhibitory neuron",
                "062 STR D2 Gaba": "Inhibitory neuron",
                "064 STR-PAL Chst9 Gaba": "Inhibitory neuron",
                "042 OB-out Frmd7 Gaba": "Inhibitory neuron",
                "039 OB Meis2 Thsd7b Gaba": "Inhibitory neuron",
                "080 CEA-AAA-BST Six3 Sp9 Gaba": "Inhibitory neuron",
                "051 Pvalb chandelier Gaba": "Inhibitory neuron",
                "044 OB Dopa-Gaba": "Inhibitory neuron",
                "045 OB-STR-CTX Inh IMN": "Inhibitory neuron",
                "065 IA Mgp Gaba": "Inhibitory neuron",
                "063 STR D1 Sema5a Gaba": "Inhibitory neuron",
                "048 RHP-COA Ndnf Gaba": "Inhibitory neuron",
                # "NN" suffix → non‐neuronal
                "327 Oligo NN": "Oligodendrocyte",
                "326 OPC NN": "Oligodendrocyte precursor cell (OPC)",
                "333 Endo NN": "Endothelial cell",
                "332 SMC NN": "Smooth muscle cell",
                "330 VLMC NN": "Vascular smooth muscle cell",
                "319 Astro-TE NN": "Astrocyte",
                "318 Astro-NT NN": "Astrocyte",
                "338 Lymphoid NN": "Lymphoid cell",
                "331 Peri NN": "Pericyte",
                "334 Microglia NN": "Microglia",
                "335 BAM NN": "Macrophage",
                "329 ABC NN": "Basal cell",
                # missing / fallback
                None: "Unclassified",
            }

            adata_sub.obs["subclass"] = np.array(
                self.slice_data_loader.gene_exp_model.get_label_from_token(
                    adata_sampled.obs["token"].tolist()
                )
            )

            adata_sub.obs["cell_type"] = adata_sub.obs["subclass"].map(new2existing)
            adata_sub.obs["cell_type"] = adata_sub.obs["cell_type"].fillna(
                "Unclassified"
            )

            cols = ["cell_type", "organ"]
            mask = adata_sub.obs[cols].notnull().all(axis=1)
            adata_sub = adata_sub[mask].copy()

            meta_info = torch.load(
                "/work/magroup/skrieger/scMulan/Tutorials/scMulan/utils/meta_info.pt"
            )
            sc.pp.normalize_total(adata_sub, target_sum=1e4)
            sc.pp.log1p(adata_sub)
            adata_sub.var.index = [g.upper() for g in adata_sub.var.index]
            adata_sub2 = adata_sub[
                :, adata_sub.var_names.isin(meta_info["gene_set"])
            ].copy()
            meta_info["gene_set"] = adata_sub2.var_names.tolist()
            print(adata_sub2.X.shape)

            # adata_sub2=GeneSymbolUniform(input_adata=adata_sub2)
            print(adata_sub2.X.shape)
            new_tokens = ["<x>", "<y>", "<z>"]

            # meta_info['token_set'].extend(new_tokens)   ##EXtend set

            tokenizer = scMulanTokenizer(meta_info["token_set"])
            ids, vals = generate_prompt_for_cg(0, adata_sub.obs, meta_info, tokenizer)

            ckp_path = "/work/magroup/skrieger/tissue_generator/spencer_gentran/ckpt/ckpt_scMulan.pt"
            ckp = torch.load(ckp_path, map_location="cpu")
            gptconf = MulanConfig(**ckp["model_args"])
            bin_edges = compute_global_bin_edges(
                adata_sub2, adata_sub2.var_names, gptconf.expression_level
            )
            ckp["model_args"]["bin_edges"] = bin_edges
            gptconf = MulanConfig(**ckp["model_args"])

            ModelClass = scMulanModel
            model = ModelClass(gptconf)
            model.load_state_dict(ckp["model"], strict=False)
            model.eval()
            model.hidden_dim = ckp["model_args"]["n_embd"]

            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)

            scml = scMulan(
                adata_sub2,
                meta_info,
                tokenizer,
                10,
                model=model.to("cuda"),
                bin_edges=bin_edges,
            )

            rows = []
            for id in tqdm.tqdm(list(range(len(adata_sub)))):
                # row, gt, nv, gen_seq, gen_vals_binned, gen_vals
                row = scml.generate_cell_genesis(
                    idx=id,
                    max_new_tokens=500,
                    top_k=5,
                )
                rows.append(row)

            rows = np.array(rows)
            obs = adata_sub.obs.copy()

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