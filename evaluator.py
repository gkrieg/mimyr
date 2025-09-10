import torch

from metrics import soft_accuracy, soft_correlation, neighborhood_enrichment, soft_f1, weighted_l1_distance, delauney_colocalization
class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, predicted_adata, target_adata, sample=100):
        results = {}

        if "weighted_l1_distance" in self.config["metrics"]:
            for k in [5, 10, 20]:
                tvd=weighted_l1_distance(target_adata.obsm["aligned_spatial"],predicted_adata.obsm["spatial"],k=k, sample=sample)
                print("weighted_l1_distance variation distance @",k,":",tvd)
                results[f"weighted_l1_distance@{k}"]=tvd

        if "soft_accuracy" in self.config["metrics"]:
            for k in [5, 10, 20]:
                sa=soft_accuracy(target_adata.obs["token"].to_numpy().tolist(),target_adata.obsm["aligned_spatial"],predicted_adata.obs["token"].tolist(),predicted_adata.obsm["spatial"],k=k,sample=sample)
                print("soft accuracy @",k,":",sa)
                results[f"soft_accuracy@{k}"]=sa

        if "delauney_colocalization" in self.config["metrics"]:
            dc=delauney_colocalization(target_adata,target_adata.obsm["aligned_spatial"],predicted_adata,predicted_adata.obsm["spatial"],sample=sample)
            print("delauney colocalization :",dc)
            results[f"delauney_colocalization"]=dc
            

        if "neighborhood_enrichment" in self.config["metrics"]:
            for k in [5, 10, 20]:
                ne=neighborhood_enrichment(target_adata.obs["token"].to_numpy().tolist(),target_adata.obsm["aligned_spatial"],predicted_adata.obs["token"].tolist(),predicted_adata.obsm["spatial"],k=k)
                print("neighborhood enrichment @",k,":",ne)
                results[f"neighborhood_enrichment@{k}"]=ne

        if "soft_correlation" in self.config["metrics"]:
            for k in [5, 10, 20]:
                sc=soft_correlation(target_adata,target_adata.obsm["aligned_spatial"],predicted_adata,predicted_adata.obsm["spatial"],k=k,sample=sample)
                print("soft correlation @",k,":",sc)
                results[f"soft_correlation@{k}"]=sc

        # F1
        if "soft_f1" in self.config["metrics"]:
            for k in [5, 10, 20]:
                sp=soft_f1(target_adata,
                                  target_adata.obsm["aligned_spatial"],
                                  predicted_adata,
                                  predicted_adata.obsm["spatial"],k=k,sample=sample)[0]
                print("soft f1 @",k,":",sp)
                results[f"soft_f1@{k}"]=sp
        return results