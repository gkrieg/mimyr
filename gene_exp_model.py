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

class GeneExpModel(nn.Module):
    def __init__(self, slices, use_subclass=False):
        super(GeneExpModel, self).__init__()
        self.slices = slices
        self.use_subclass = use_subclass
        self.num_tokens=0        
        

        # bandwidth = 0.05  # tune this!

        k = 50  # Number of neighbors to use for density estimate
        bandwidth = 0.05  # For entropy KDE
        d = 3  # Dimensionality of spatial coordinates

        volume_unit_ball = np.pi ** (d / 2) / gamma(d / 2 + 1)

        for slice in slices:
            positions = slice.obsm["aligned_spatial"]
            gt_tree = cKDTree(positions)

            densities = {1:[],5:[],10:[],15:[],20:[]}

            spreads = []
            entropies = []

            k_entropy = 30  # or any number you want
            for i, pos in tqdm.tqdm(list(enumerate(positions))):
                # === DENSITIES === (unchanged from your version)
                for k in [1, 5, 10, 15, 20]:
                    distances, _ = gt_tree.query(pos, k=k+1)
                    r_k = distances[-1]
                    densities[k].append(1 / (r_k + 1e-12))

                # === SPREAD === (still using fixed radius)
                neighbors_idx = gt_tree.query_ball_point(pos, 0.05)
                neighborhood = positions[neighbors_idx]
                if len(neighborhood) > 1:
                    centered = neighborhood - neighborhood.mean(axis=0)
                    cov = np.cov(centered.T)
                    spread = np.trace(cov)
                else:
                    spread = 0.0
                spreads.append(spread)

                # === ENTROPY using k nearest neighbors ===
                ds, knn_idx = gt_tree.query(pos, k=k_entropy + 1)  # includes self
                neighborhood_knn = positions[knn_idx[1:]]/ds[-1]  # exclude the point itself

                if len(neighborhood_knn) > 2:
                    kde = KernelDensity(bandwidth=0.5)
                    kde.fit(neighborhood_knn)
                    log_probs = kde.score_samples(neighborhood_knn)
                    entropy = -np.mean(log_probs)
                else:
                    entropy = np.nan
                entropies.append(entropy)

            slice.obs["density1"] = np.array(densities[1])
            slice.obs["density5"] = np.array(densities[5])
            slice.obs["density10"] = np.array(densities[10])
            slice.obs["density15"] = np.array(densities[15])
            slice.obs["density20"] = np.array(densities[20])

            # slice.obs["spread"] = np.array(spreads)
            slice.obs["entropy"] = np.array(entropies)
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
        if self.use_subclass:
            concatenated_slices=ad.concat(self.slices)
            unique_subclasses = concatenated_slices.obs["class"].unique()
            self.num_tokens=len(unique_subclasses)
            subclass_to_id = {subclass: i for i, subclass in enumerate(unique_subclasses)}
            self.subclass_to_id=subclass_to_id
            concatenated_slices.obs["token"] = concatenated_slices.obs["class"].map(subclass_to_id)

            gene_exp_dict={i:concatenated_slices[concatenated_slices.obs["token"]==i].X.mean(0) for i in range(len(unique_subclasses))}            
            # self.gene_exp_to_token = {concatenated_slices.X[i].sum():concatenated_slices.obs["token"][i] for i in range(len(concatenated_slices))}
            row_sums = np.array(concatenated_slices.X.sum(axis=1)).flatten()  # Convert to 1D NumPy array
            tokens = concatenated_slices.obs["token"].values  # Convert to NumPy array if needed
            self.gene_exp_to_token = dict(zip(row_sums.flatten(), tokens))  # Use vectorized zip

            self.token_to_gene_exp = {i:gene_exp_dict[i] for i in range(len(unique_subclasses))}
            self.concatenated_slices=concatenated_slices
        else:
            pass
    
    def get_gene_exp_from_token(self, tokens):
        return [self.token_to_gene_exp[token] for token in tokens]

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