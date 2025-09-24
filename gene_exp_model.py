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
import sys
sys.path.append("generative_transformer")

from scMulan import generate_prompt_for_cg

from utils.hf_tokenizer import scMulanTokenizer

from generative_transformer.scMulan import compute_global_bin_edges
import scMulan
import torch
from model.model import MulanConfig, scMulanModel
from scMulan import scMulan
import pickle as pkl

class GeneExpModel(nn.Module):
    def __init__(self, slices, label="subclass"):
        super(GeneExpModel, self).__init__()
        self.slices = slices
        self.label = label
        self.num_tokens=0        
        

        # bandwidth = 0.05  # tune this!

        k = 50  # Number of neighbors to use for density estimate
        bandwidth = 0.05  # For entropy KDE
        d = 3  # Dimensionality of spatial coordinates

        volume_unit_ball = np.pi ** (d / 2) / gamma(d / 2 + 1)

        # for slice in slices:
        #     positions = slice.obsm["aligned_spatial"]
        #     gt_tree = cKDTree(positions)

        #     densities = {1:[],5:[],10:[],15:[],20:[]}

        #     spreads = []
        #     entropies = []

        #     k_entropy = 30  # or any number you want
        #     for i, pos in tqdm.tqdm(list(enumerate(positions))):
        #         # === DENSITIES === (unchanged from your version)
        #         for k in [1, 5, 10, 15, 20]:
        #             distances, _ = gt_tree.query(pos, k=k+1)
        #             r_k = distances[-1]
        #             densities[k].append(1 / (r_k + 1e-12))

        #         # === SPREAD === (still using fixed radius)
        #         neighbors_idx = gt_tree.query_ball_point(pos, 0.05)
        #         neighborhood = positions[neighbors_idx]
        #         if len(neighborhood) > 1:
        #             centered = neighborhood - neighborhood.mean(axis=0)
        #             cov = np.cov(centered.T)
        #             spread = np.trace(cov)
        #         else:
        #             spread = 0.0
        #         spreads.append(spread)

        #         # === ENTROPY using k nearest neighbors ===
        #         ds, knn_idx = gt_tree.query(pos, k=k_entropy + 1)  # includes self
        #         neighborhood_knn = positions[knn_idx[1:]]/ds[-1]  # exclude the point itself

        #         if len(neighborhood_knn) > 2:
        #             kde = KernelDensity(bandwidth=0.5)
        #             kde.fit(neighborhood_knn)
        #             log_probs = kde.score_samples(neighborhood_knn)
        #             entropy = -np.mean(log_probs)
        #         else:
        #             entropy = np.nan
        #         entropies.append(entropy)

        #     slice.obs["density1"] = np.array(densities[1])
        #     slice.obs["density5"] = np.array(densities[5])
        #     slice.obs["density10"] = np.array(densities[10])
        #     slice.obs["density15"] = np.array(densities[15])
        #     slice.obs["density20"] = np.array(densities[20])

        #     # slice.obs["spread"] = np.array(spreads)
        #     slice.obs["entropy"] = np.array(entropies)
        # for slice in slices:
        #     aligned = slice.obsm["aligned_spatial"]
        #     gt_tree = cKDTree(aligned)
        #     pca_features = []
            

        #     for i, pos in tqdm.tqdm(list(enumerate(aligned))):
        #         neighbors_idx = gt_tree.query_ball_point(pos, r=0.05)
        #         if len(neighbors_idx) < 2:
        #             # not enough points to define PCA
        #             direction = np.zeros(aligned.shape[1])
        #             component = 0.0
        #         else:
        #             neighborhood = aligned[neighbors_idx]
        #             neighborhood_centered = neighborhood - neighborhood.mean(axis=0)
        #             pca = PCA(n_components=1)
        #             pca.fit(neighborhood_centered)
        #             direction = pca.components_[0]  # shape (3,)
        #             component = np.dot(pos - neighborhood.mean(axis=0), direction)

        #         # store concatenated: [dir_x, dir_y, dir_z, projected_value]
        #         pca_features.append(np.concatenate([direction, [component]]))

        #     slice.obsm["pca"] = np.array(pca_features)

    def fit(self):
        concatenated_slices=ad.concat(self.slices)
        unique_subclasses = concatenated_slices.obs[self.label].unique()
        self.num_tokens=len(unique_subclasses)
        try:
            self.id_to_subclass = pkl.load(open("id_to_subclass.pkl","rb"))
            subclass_to_id = {v:k for k,v in self.id_to_subclass.items()}
        except:
            subclass_to_id = {subclass: i for i, subclass in enumerate(unique_subclasses)}
            self.id_to_subclass = {i:subclass for i, subclass in enumerate(unique_subclasses)}
            print("Recreating token map")
        self.subclass_to_id=subclass_to_id
        concatenated_slices.obs["token"] = concatenated_slices.obs[self.label].map(subclass_to_id)

        self.concatenated_slices=concatenated_slices

        return
        ### takes a long time, so removing unless averaging is being used

        gene_exp_dict={i:concatenated_slices[concatenated_slices.obs["token"]==i].X.mean(0) for i in range(len(unique_subclasses))}  
        print(gene_exp_dict[0].shape)
        # self.gene_exp_to_token = {concatenated_slices.X[i].sum():concatenated_slices.obs["token"][i] for i in range(len(concatenated_slices))}
        row_sums = np.array(concatenated_slices.X.sum(axis=1)).flatten()  # Convert to 1D NumPy array
        tokens = concatenated_slices.obs["token"].values  # Convert to NumPy array if needed
        self.gene_exp_to_token = dict(zip(row_sums.flatten(), tokens))  # Use vectorized zip

        self.token_to_gene_exp = {i:gene_exp_dict[i] for i in range(len(unique_subclasses))}
        self.concatenated_slices=concatenated_slices
    
    def get_gene_exp_from_token(self, tokens):
        return [self.token_to_gene_exp[token] for token in tokens]

    def get_label_from_token(self, tokens):
        return [self.id_to_subclass[token] for token in tokens]

    def get_token_from_gene_exp(self, gene_exps):
        return [self.gene_exp_to_token[gene_exp] for gene_exp in gene_exps]

    def get_tokenized_slices(self):
        slices_new=[]
        c=0
        for slice in self.slices:
            slices_new.append(self.concatenated_slices[c:c+len(slice)])
            c+=len(slice)
        return slices_new

    def get_gene_exp_from_transformer(self,adata_sub):

        coord_bins = {}
        n_bins = 10  # e.g. 10

        adata_sub.obs['x']=adata_sub.obsm['aligned_spatial'][:,0]
        adata_sub.obs['y']=adata_sub.obsm['aligned_spatial'][:,1]
        adata_sub.obs['z']=adata_sub.obsm['aligned_spatial'][:,2]

        adata_sub.obs['organ'] = 'Brain'



        region_map = {
            'SS-GU-VISC':  'Cerebral cortex',  # somatosensory gustatory/visceral area
            'PL-ILA-ORB':  'Cerebral cortex',  # prelimbic, infralimbic, orbital prefrontal
            'TEa-PERI-ECT':'Cerebral cortex',  # temporal association & perirhinal
            'MOp':         'Cerebral cortex',  # primary motor cortex
            'VIS':         'Cerebral cortex',  # visual cortex
            'VIS-PTLp':    'Cerebral cortex',  # posterior lateral visual
            'SSp':         'Cerebral cortex',  # primary somatosensory cortex
            'MO-FRP':      'Cerebral cortex',  # frontal pole motor
            'AI':          'Cerebral cortex',  # agranular insular cortex
            'AUD':         'Cerebral cortex',  # auditory cortex

            'ACA':         'Cingulate cortex', # anterior cingulate area
            'RSP':         'Cingulate cortex', # retrosplenial cortex

            np.nan:        'Unclassified'     # missing values
        }
        # adata_sub.obs['region'] = adata_sub.obs['region_of_interest_acronym'].map(region_map)
        # adata_sub.obs['region'] = adata_sub.obs['region'].fillna('Unclassified')
        new2existing = {
            # glutamatergic → Excitatory neuron
            '006 L4/5 IT CTX Glut':                      'Excitatory neuron',
            '030 L6 CT CTX Glut':                        'Excitatory neuron',
            '029 L6b CTX Glut':                         'Excitatory neuron',
            '032 L5 NP CTX Glut':                       'Excitatory neuron',
            '005 L5 IT CTX Glut':                       'Excitatory neuron',
            '007 L2/3 IT CTX Glut':                     'Excitatory neuron',
            '022 L5 ET CTX Glut':                       'Excitatory neuron',
            '021 L4 RSP-ACA Glut':                      'Excitatory neuron',
            '004 L6 IT CTX Glut':                       'Excitatory neuron',
            '020 L2/3 IT RSP Glut':                     'Excitatory neuron',
            '001 CLA-EPd-CTX Car3 Glut':                'Excitatory neuron',
            '003 L5/6 IT TPE-ENT Glut':                 'Excitatory neuron',
            '028 L6b/CT ENT Glut':                      'Excitatory neuron',
            '002 IT EP-CLA Glut':                       'Excitatory neuron',
            '025 CA2-FC-IG Glut':                       'Excitatory neuron',
            '009 L2/3 IT PIR-ENTl Glut':                'Excitatory neuron',
            '010 IT AON-TT-DP Glut':                    'Excitatory neuron',
            '036 HPF CR Glut':                          'Excitatory neuron',
            '008 L2/3 IT ENT Glut':                     'Excitatory neuron',
            '027 L6b EPd Glut':                         'Excitatory neuron',
            '114 COAa-PAA-MEA Barhl2 Glut':             'Excitatory neuron',
            '018 L2 IT PPP-APr Glut':                   'Excitatory neuron',
            '035 OB Eomes Ms4a15 Glut':                 'Excitatory neuron',
            '262 Pineal Crx Glut':                      'Excitatory neuron',
            '034 NP PPP Glut':                          'Excitatory neuron',
            '019 L2/3 IT PPP Glut':                     'Excitatory neuron',
            '033 NP SUB Glut':                          'Excitatory neuron',
            '115 MS-SF Bsx Glut':                       'Excitatory neuron',
            '016 CA1-ProS Glut':                        'Excitatory neuron',

            # GABAergic → Inhibitory neuron
            '053 Sst Gaba':                             'Inhibitory neuron',
            '050 Lamp5 Lhx6 Gaba':                     'Inhibitory neuron',
            '046 Vip Gaba':                             'Inhibitory neuron',
            '052 Pvalb Gaba':                           'Inhibitory neuron',
            '049 Lamp5 Gaba':                          'Inhibitory neuron',
            '047 Sncg Gaba':                           'Inhibitory neuron',
            '056 Sst Chodl Gaba':                      'Inhibitory neuron',
            '041 OB-in Frmd7 Gaba':                    'Inhibitory neuron',
            '066 NDB-SI-ant Prdm12 Gaba':              'Inhibitory neuron',
            '061 STR D1 Gaba':                         'Inhibitory neuron',
            '062 STR D2 Gaba':                         'Inhibitory neuron',
            '064 STR-PAL Chst9 Gaba':                  'Inhibitory neuron',
            '042 OB-out Frmd7 Gaba':                   'Inhibitory neuron',
            '039 OB Meis2 Thsd7b Gaba':                'Inhibitory neuron',
            '080 CEA-AAA-BST Six3 Sp9 Gaba':           'Inhibitory neuron',
            '051 Pvalb chandelier Gaba':               'Inhibitory neuron',
            '044 OB Dopa-Gaba':                        'Inhibitory neuron',
            '045 OB-STR-CTX Inh IMN':                  'Inhibitory neuron',
            '065 IA Mgp Gaba':                         'Inhibitory neuron',
            '063 STR D1 Sema5a Gaba':                  'Inhibitory neuron',
            '048 RHP-COA Ndnf Gaba':                   'Inhibitory neuron',

            # "NN" suffix → non‐neuronal
            '327 Oligo NN':                            'Oligodendrocyte',
            '326 OPC NN':                              'Oligodendrocyte precursor cell (OPC)',
            '333 Endo NN':                             'Endothelial cell',
            '332 SMC NN':                              'Smooth muscle cell',
            '330 VLMC NN':                             'Vascular smooth muscle cell',
            '319 Astro-TE NN':                         'Astrocyte',
            '318 Astro-NT NN':                         'Astrocyte',
            '338 Lymphoid NN':                         'Lymphoid cell',
            '331 Peri NN':                             'Pericyte',
            '334 Microglia NN':                        'Microglia',
            '335 BAM NN':                              'Macrophage',
            '329 ABC NN':                              'Basal cell',

            # missing / fallback
            None:                                      'Unclassified',
        }

        adata_sub.obs['cell_type'] = adata_sub.obs['subclass'].map(new2existing)
        adata_sub.obs['cell_type'] = adata_sub.obs['cell_type'].fillna('Unclassified')




        # for coord in ('x','y','z'):
        #     vals_full = adata_sub.obs[coord].values.astype(float)
        #     vals = adata_sub.obs[coord].dropna().values.astype(float)
        #     coord_bins[coord] = np.linspace(vals.min(), vals.max(), n_bins)
        #     edges   = coord_bins[coord]
        #     bin_idxs = np.digitize(vals_full, edges, right=True)
        #     adata_sub.obs[f'<{coord}>'] = bin_idxs
        # cols = ['x','y','z','cell_type','organ']  ##extend token set


        cols = ['cell_type','organ']
        mask = adata_sub.obs[cols].notnull().all(axis=1)
        adata_sub = adata_sub[mask].copy()


        meta_info = torch.load('/work/magroup/skrieger/scMulan/Tutorials/scMulan/utils/meta_info.pt')
        sc.pp.normalize_total(adata_sub,target_sum=1e4)
        sc.pp.log1p(adata_sub)
        adata_sub.var.index=[g.upper() for g in adata_sub.var.index]
        adata_sub2=adata_sub[:,adata_sub.var_names.isin(meta_info["gene_set"])].copy()


        new_tokens = ["<x>", "<y>", "<z>"]

        # meta_info['token_set'].extend(new_tokens)   ##EXtend set

        tokenizer = scMulanTokenizer(meta_info['token_set'])
        ids, vals = generate_prompt_for_cg(0, adata_sub.obs, meta_info, tokenizer)


        ckp_path = '/work/magroup/skrieger/tissue_generator/spencer_gentran/ckpt/ckpt_scMulan.pt'
        ckp = torch.load(ckp_path, map_location='cpu')
        gptconf = MulanConfig(**ckp['model_args'])
        bin_edges=compute_global_bin_edges(adata_sub, adata_sub2.var_names,gptconf.expression_level)
        ckp['model_args']["bin_edges"]=bin_edges
        gptconf = MulanConfig(**ckp['model_args'])


        ModelClass = scMulanModel
        model = ModelClass(gptconf)
        model.load_state_dict(ckp['model'], strict=False)
        model.eval()
        model.hidden_dim = ckp['model_args']['n_embd']

        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

        scml = scMulan(adata_sub2,meta_info,tokenizer,10,model=model.to("cuda"),bin_edges=bin_edges)

        row, gt, nv, gen_seq, gen_vals_binned, gen_vals = scml.generate_cell_genesis(
                    idx=0,
                    max_new_tokens= 500,
                    top_k= 5,
                )
        

        return row





if __name__=="__main__":
    test_data = ad.read("/work/magroup/skrieger/tissue_generator/quantized_slices/subclass_z1_d338_0_rotated/sec_40.h5ad")
    model = GeneExpModel([],test_data,[],use_subclass=True)
    model.fit()
    print(model.get_token_from_gene_exp([test_data.X[0].sum()]))
    print(model.get_gene_exp_from_token([0]))