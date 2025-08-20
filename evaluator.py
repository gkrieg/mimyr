import torch

from metrics import soft_accuracy, soft_correlation, neighborhood_enrichment, soft_precision
class Evaluator:
    def evaluate(self, predicted_adata, target_adata, sample=100):
        results = {}
        for k in [5, 10, 20]:
            sa=soft_accuracy(target_adata.obs["token"].to_numpy().tolist(),target_adata.obsm["aligned_spatial"],predicted_adata.obs["token"].tolist(),predicted_adata.obsm["spatial"],k=k,sample=sample)
            print("soft accuracy @",k,":",sa)
            results[f"soft_accuracy@{k}"]=sa

        for k in [5, 10, 20]:
            ne=neighborhood_enrichment(target_adata.obs["token"].to_numpy().tolist(),target_adata.obsm["aligned_spatial"],predicted_adata.obs["token"].tolist(),predicted_adata.obsm["spatial"],k=k)
            print("neighborhood enrichment @",k,":",ne)
            results[f"neighborhood_enrichment@{k}"]=ne


        for k in [5, 10, 20]:

            sc=soft_correlation(target_adata,target_adata.obsm["aligned_spatial"],predicted_adata,predicted_adata.obsm["spatial"],k=k,sample=sample)
            print("soft correlation @",k,":",sc)
            results[f"soft_correlation@{k}"]=sc

        for k in [5, 10, 20]:
            sp=soft_precision(target_adata,
                              target_adata.obsm["aligned_spatial"],
                              predicted_adata,
                              predicted_adata.obsm["spatial"],k=k,sample=sample)


            print("soft precision @",k,":",sp)
            results[f"soft_precision@{k}"]=sp
        return results

        