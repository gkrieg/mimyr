import torch
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


### --------------------- CORE FUNCTIONS ---------------------

def generate_anndata_from_samples(region_model, xyz, device="cuda", sample_from_probs=False):
    region_model.eval().to(device)
    with torch.no_grad():
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).to(device)
        probs = torch.softmax(region_model.model(xyz_tensor), dim=1).cpu().numpy()
        preds = np.array([np.random.choice(len(p), p=p) for p in probs]) if sample_from_probs else np.argmax(probs, axis=1)
    adata = ad.AnnData(X=np.zeros((xyz.shape[0], 1)))
    adata.obsm["spatial"] = xyz[:, :3].cpu().numpy()
    adata.obs["token"] = preds
    return adata, preds


def assign_shared_colors(adatas, color_key="token", palette="tab10"):
    all_labels = pd.Categorical(np.concatenate([adata.obs[color_key] for adata in adatas])).categories
    colors = [mcolors.to_hex(c) for c in sns.color_palette(palette, n_colors=len(all_labels))]
    for adata in adatas:
        adata.obs[color_key] = pd.Categorical(adata.obs[color_key], categories=all_labels)
        adata.uns[f"{color_key}_colors"] = colors


def plot_spatial_with_palette(
    adata, 
    color_key="token", 
    spot_size=0.001, 
    size=10, 
    scale_factor=1, 
    show_legend=False, 
    figsize=(10, 10), 
    save=None
):
    with plt.rc_context({"figure.figsize": figsize}):
        sc.pl.spatial(
            adata,
            color=color_key,
            spot_size=spot_size,
            size=size,
            scale_factor=scale_factor,
            legend_loc="right margin" if show_legend else None,
            save=save
        )


### --------------------- EVALUATION FUNCTIONS ---------------------

def generate_topk_miss_anndata(region_model, xyz, true_labels, k=20, which_pred=1, device="cuda"):
    region_model.eval().to(device)
    with torch.no_grad():
        probs = torch.softmax(region_model.model(torch.tensor(xyz, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        topk = np.argsort(probs, axis=1)[:, -k:]
    missed = np.array([t not in tk for t, tk in zip(true_labels, topk)])
    xyz_miss = xyz[missed]
    preds = np.argsort(probs[missed], axis=1)[:, -which_pred]
    adata = ad.AnnData(X=np.zeros((xyz_miss.shape[0], 1)))
    adata.obsm["spatial"] = xyz_miss.cpu().numpy()
    adata.obs["token"] = preds
    adata.obs["true_label"] = true_labels[missed].astype(int)
    return adata


def generate_topk_correctness_anndata(region_model, xyz, true_labels, k=20, device="cuda"):
    region_model.eval().to(device)
    with torch.no_grad():
        probs = torch.softmax(region_model.model(torch.tensor(xyz, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        topk = np.argsort(probs, axis=1)[:, -k:]
        preds = np.argmax(probs, axis=1)
    missed = np.array([t not in tk for t, tk in zip(true_labels, topk)])
    adata = ad.AnnData(X=np.zeros((xyz.shape[0], 1)))
    adata.obsm["spatial"] = xyz.cpu().numpy()
    adata.obs["true_label"] = true_labels.astype(int)
    adata.obs["pred_label"] = preds
    adata.obs["missed"] = missed
    return adata


def plot_correctness(adata, spot_size=0.005):
    coords = adata.obsm["spatial"]
    missed = adata.obs["missed"]
    correct = coords[~missed]
    incorrect = coords[missed]
    print("Acc:", len(correct) / len(coords))
    plt.figure(figsize=(6, 6))
    plt.scatter(correct[:, 0], -correct[:, 1], c="white", s=spot_size * 1e4, edgecolors="k", linewidths=0.02)
    plt.scatter(incorrect[:, 0], -incorrect[:, 1], c="red", s=spot_size * 1e4, edgecolors="k", linewidths=0.02)
    plt.axis("off")
    plt.title("Red = Incorrect | White = Correct (Top-k)")
    plt.show()


def plot_soft_scores(adata, spot_size=0.005, cmap="RdYlGn"):
    coords = adata.obsm["spatial"]
    scores = adata.obs["soft_score"]
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(coords[:, 0], -coords[:, 1], c=scores, cmap=cmap,
                     s=spot_size * 1e4, edgecolors="k", linewidths=0.02)
    plt.colorbar(sc, label="Soft Accuracy Score")
    plt.axis("off")
    plt.title("Per-cell Soft Accuracy")
    plt.show()
