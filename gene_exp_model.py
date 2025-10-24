from sklearn.decomposition import PCA
import math
import pickle as pkl

import numpy as np
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial import cKDTree
from scipy.special import gamma
from sklearn.neighbors import KernelDensity

from generative_transformer import generate_prompt_for_cg
from generative_transformer.scMulan import compute_global_bin_edges
from generative_transformer.model.model import MulanConfig, scMulanModel
from generative_transformer.utils.hf_tokenizer import scMulanTokenizer

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

    

if __name__=="__main__":
    test_data = ad.read("/work/magroup/skrieger/tissue_generator/quantized_slices/subclass_z1_d338_0_rotated/sec_40.h5ad")
    model = GeneExpModel([],test_data,[],use_subclass=True)
    model.fit()
    print(model.get_token_from_gene_exp([test_data.X[0].sum()]))
    print(model.get_gene_exp_from_token([0]))