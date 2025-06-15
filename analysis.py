import torch
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm

# ### --------------------- CORE FUNCTIONS ---------------------
# def generate_anndata_from_samples(region_model, xyz, device="cuda", sample_from_probs=False,
#                                   use_conditionals=False, xyz_labels=None, num_classes=None,
#                                   gibbs=False, n_iter=5):
#     """
#     xyz: [N, D] numpy array of coordinates (must include 3 spatial dims in [:, :3])
#     xyz_labels: [N] array of known tokens to use for conditioning (initial labels if gibbs=True)
#     """
#     region_model.eval().to(device)

#     with torch.no_grad():
#         xyz_tensor = torch.tensor(xyz, dtype=torch.float32).to(device)

#         if use_conditionals:
#             if num_classes is None:
#                 raise ValueError("num_classes must be provided when use_conditionals=True")
            
#             # Initial labels for gibbs
#             if gibbs:
#                 if isinstance(xyz, torch.Tensor):
#                     xyz = xyz.detach().cpu().numpy()

#                 if xyz_labels is None:
#                     # initialize randomly if not provided
#                     xyz_labels = np.random.randint(0, num_classes, size=xyz.shape[0])
#                 else:
#                     xyz_labels = np.array(xyz_labels)

#                 from sklearn.neighbors import NearestNeighbors
#                 nbrs = NearestNeighbors(n_neighbors=6).fit(xyz[:, :3])
#                 _, indices = nbrs.kneighbors(xyz[:, :3])
#                 neighbor_indices = indices[:, 1:]  # [N, 5]

#                 for _ in tqdm.trange(n_iter):
#                     for i in range(len(xyz_labels)):
#                         # Gather current neighbor indices for i
#                         neighbors = neighbor_indices[i]
                        
#                         # Use most up-to-date labels for neighbors
#                         neighbor_tokens = xyz_labels[neighbors]
                        
#                         # One-hot encode and sum over neighbors
#                         neighbor_oh = torch.nn.functional.one_hot(
#                             torch.tensor(neighbor_tokens), num_classes=num_classes
#                         ).float().sum(dim=0).to(device)  # [C]

#                         # Construct input: [xyz_i | neighbor_oh]
#                         input_tensor = torch.cat([xyz_tensor[i], neighbor_oh], dim=0).unsqueeze(0)  # [1, D+C]

#                         # Predict distribution and sample/update label
#                         probs = torch.softmax(region_model.model(input_tensor), dim=1).squeeze(0).cpu().numpy()
#                         if sample_from_probs:
#                             xyz_labels[i] = np.random.choice(num_classes, p=probs)
#                         else:
#                             xyz_labels[i] = np.argmax(probs)
#                 preds = xyz_labels
#             else:
#                 if isinstance(xyz, torch.Tensor):
#                     xyz = xyz.detach().cpu().numpy()

#                 if xyz_labels is None:
#                     raise ValueError("xyz_labels must be provided when use_conditionals=True and gibbs=False")

#                 from sklearn.neighbors import NearestNeighbors
#                 nbrs = NearestNeighbors(n_neighbors=6).fit(xyz[:, :3])
#                 _, indices = nbrs.kneighbors(xyz[:, :3])
#                 neighbor_indices = indices[:, 1:]

#                 neighbor_tokens = np.array(xyz_labels)[neighbor_indices]  # [N, 5]
#                 neighbor_oh = torch.nn.functional.one_hot(
#                     torch.tensor(neighbor_tokens), num_classes=num_classes
#                 ).float().sum(-2).to(device)

#                 input_tensor = torch.cat([xyz_tensor, neighbor_oh], dim=-1)
#                 probs = torch.softmax(region_model.model(input_tensor), dim=1).cpu().numpy()
#                 preds = np.array([
#                     np.random.choice(len(p), p=p) if sample_from_probs else np.argmax(p)
#                     for p in probs
#                 ])
#         else:
#             input_tensor = xyz_tensor
#             probs = torch.softmax(region_model.model(input_tensor), dim=1).cpu().numpy()
#             preds = np.array([
#                 np.random.choice(len(p), p=p) if sample_from_probs else np.argmax(p)
#                 for p in probs
#             ])

#     if isinstance(xyz, torch.Tensor):
#         xyz = xyz.detach().cpu().numpy()

#     adata = ad.AnnData(X=np.zeros((xyz.shape[0], 1)))
#     adata.obsm["spatial"] = xyz[:, :3]
#     adata.obs["token"] = preds
#     return adata, preds



def generate_anndata_from_samples(region_model, xyz, device="cuda", sample_from_probs=False,
                                  use_conditionals=False, xyz_labels=None, num_classes=None,
                                  gibbs=False, n_iter=5, use_budget=False, graph_smooth=False):
    """
    xyz: [N, D] numpy array of coordinates (must include 3 spatial dims in [:, :3])
    xyz_labels: [N] array of known tokens to use for conditioning (initial labels if gibbs=True)
    use_budget: when True and use_conditionals=False, enforces budget-based sampling per group
    graph_smooth: when True and use_conditionals=False, applies 1-step GCN-like smoothing before sampling
    """
    import torch
    import numpy as np
    import tqdm
    import anndata as ad
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    import random

    region_model.eval().to(device)

    with torch.no_grad():
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).to(device)

        if use_conditionals:
            if num_classes is None:
                raise ValueError("num_classes must be provided when use_conditionals=True")

            if gibbs:
                if isinstance(xyz, torch.Tensor):
                    xyz = xyz.detach().cpu().numpy()

                if xyz_labels is None:
                    xyz_labels = np.random.randint(0, num_classes, size=xyz.shape[0])
                else:
                    xyz_labels = np.array(xyz_labels)

                nbrs = NearestNeighbors(n_neighbors=6).fit(xyz[:, :3])
                _, indices = nbrs.kneighbors(xyz[:, :3])
                neighbor_indices = indices[:, 1:]  # [N, 5]

                for _ in tqdm.trange(n_iter):
                    for i in range(len(xyz_labels)):
                        neighbors = neighbor_indices[i]
                        neighbor_tokens = xyz_labels[neighbors]

                        neighbor_oh = torch.nn.functional.one_hot(
                            torch.tensor(neighbor_tokens), num_classes=num_classes
                        ).float().sum(dim=0).to(device)  # [C]

                        input_tensor = torch.cat([xyz_tensor[i], neighbor_oh], dim=0).unsqueeze(0)  # [1, D+C]
                        probs = torch.softmax(region_model.model(input_tensor), dim=1).squeeze(0).cpu().numpy()

                        if sample_from_probs:
                            xyz_labels[i] = np.random.choice(num_classes, p=probs)
                        else:
                            xyz_labels[i] = np.argmax(probs)
                preds = xyz_labels
                probs = None
            else:
                if isinstance(xyz, torch.Tensor):
                    xyz = xyz.detach().cpu().numpy()

                if xyz_labels is None:
                    raise ValueError("xyz_labels must be provided when use_conditionals=True and gibbs=False")

                nbrs = NearestNeighbors(n_neighbors=6).fit(xyz[:, :3])
                _, indices = nbrs.kneighbors(xyz[:, :3])
                neighbor_indices = indices[:, 1:]

                neighbor_tokens = np.array(xyz_labels)[neighbor_indices]  # [N, 5]
                neighbor_oh = torch.nn.functional.one_hot(
                    torch.tensor(neighbor_tokens), num_classes=num_classes
                ).float().sum(-2).to(device)

                input_tensor = torch.cat([xyz_tensor, neighbor_oh], dim=-1)
                logits = region_model.model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.array([
                    np.random.choice(len(p), p=p) if sample_from_probs else np.argmax(p)
                    for p in probs
                ])
        else:
            input_tensor = xyz_tensor
            logits = region_model.model(input_tensor)  # [N, C]

            if graph_smooth:
                # Apply 1-hop GCN-like smoothing (average neighbor logits)
                if isinstance(xyz, torch.Tensor):
                    xyz = xyz.detach().cpu().numpy()

                nbrs = NearestNeighbors(n_neighbors=6).fit(xyz[:, :3])
                _, indices = nbrs.kneighbors(xyz[:, :3])
                neighbor_indices = indices[:, 1:]  # [N, 5]

                logits = logits.cpu()
                smoothed_logits = logits.clone()

                for i in range(len(logits)):
                    neighbors = neighbor_indices[i]
                    neighbor_logits = logits[neighbors]
                    smoothed_logits[i] = neighbor_logits.mean(dim=0)

                logits = smoothed_logits.to(device)

            probs = torch.softmax(logits, dim=1).cpu().numpy()

            if use_budget:
                from collections import defaultdict
                n_clusters = max(1, len(xyz) // 20)
                kmeans = KMeans(n_clusters=n_clusters).fit(xyz[:, :3])
                group_labels = kmeans.labels_

                preds = np.zeros(len(xyz), dtype=int)

                for g in np.unique(group_labels):
                    group_indices = np.where(group_labels == g)[0]
                    group_probs = probs[group_indices]  # shape [G, C]
                    budget = group_probs.sum(axis=0)  # shape [C]

                    remaining = list(group_indices)

                    while remaining:
                        i = random.choice(remaining)
                        p = probs[i]
                        sampled_class = np.random.choice(len(p), p=p)
                        preds[i] = sampled_class

                        budget[sampled_class] -= 1.0
                        budget = np.maximum(budget, 0.0)

                        remaining.remove(i)
            else:
                preds = np.array([
                    np.random.choice(len(p), p=p) if sample_from_probs else np.argmax(p)
                    for p in probs
                ])

    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()

    adata = ad.AnnData(X=np.zeros((xyz.shape[0], 1)))
    adata.obsm["spatial"] = xyz[:, :3]
    adata.obs["token"] = preds
    return adata, preds, probs


def homogenize(xyz, celltypes, k=5, alpha=0.9, n_iter=5, maximize=False, seed=None, probs=None):
    """
    Homogenizes a set of celltype labels by probabilistically replacing each cell's label
    with either a randomly chosen or the most frequent k-nearest neighbor's label.

    Args:
        xyz: [N, D] numpy array of spatial coordinates (use [:, :3] for spatial neighborhood)
        celltypes: [N] numpy array of integer celltype labels
        k: number of nearest neighbors to consider
        alpha: probability of keeping the original celltype
        n_iter: number of iterations to run the homogenization
        maximize: if True, replace with the most frequent label among neighbors instead of sampling
        seed: optional, for reproducibility

    Returns:
        homogenized_labels: numpy array of updated celltype labels
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter

    if seed is not None:
        np.random.seed(seed)

    xyz = np.asarray(xyz)
    labels = np.array(celltypes)

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(xyz[:, :3])
    _, indices = nbrs.kneighbors(xyz[:, :3])
    neighbor_indices = indices[:, 1:]  # skip self

    for _ in range(n_iter):
        # new_labels = labels.copy()
        for i in range(len(labels)):
            if np.random.rand() > alpha:
                neighbors = neighbor_indices[i]
                if maximize:
                    neighbor_labels = labels[neighbors]
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    labels[i] = most_common
                else:
                    neighbor = np.random.choice(neighbors)
                    temp = labels[i]
                    labels[i] = labels[neighbor]
                    labels[neighbor] = temp
        # labels = new_labels

    return labels


def gibbs_homogenize(xyz, probs, k=5, coupling=1.0, n_iter=5, seed=None):
    """
    Gibbs sampling from a pairwise UGM with unary potentials from probs and pairwise agreement potentials.

    Args:
        xyz: [N, D] spatial coordinates (only [:, :3] used)
        probs: [N, C] model marginals (probabilities for each class)
        k: number of nearest neighbors to use in the graph
        coupling: float, controls strength of neighbor agreement
        n_iter: number of full Gibbs sweeps over all nodes
        seed: for reproducibility

    Returns:
        labels: [N] array of sampled labels
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if seed is not None:
        np.random.seed(seed)

    xyz = np.asarray(xyz)
    probs = np.asarray(probs)
    N, C = probs.shape

    # Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xyz[:, :3])
    _, indices = nbrs.kneighbors(xyz[:, :3])
    neighbor_indices = indices[:, 1:]  # skip self

    # Initialize labels by sampling from marginals
    labels = np.array([np.random.choice(C, p=probs[i]) for i in range(N)])

    for _ in range(n_iter):
        for i in range(N):
            neighbors = neighbor_indices[i]
            neighbor_labels = labels[neighbors]

            # Unary log-probabilities (from model)
            logp_unary = np.log(probs[i] + 1e-12)  # [C]

            # Pairwise log potentials: +coupling for matching neighbor labels
            counts = np.bincount(neighbor_labels, minlength=C)
            logp_pairwise = coupling * counts

            # Total log probability
            logp_total = logp_unary + logp_pairwise
            logp_total -= np.max(logp_total)  # for numerical stability
            p = np.exp(logp_total)
            p /= p.sum()

            labels[i] = np.random.choice(C, p=p)

    return labels



def label_propagation(xyz, probs, k=5, entropy_threshold=0.5, n_iter=10, seed=None):
    """
    Performs label propagation where low-entropy nodes are fixed and high-entropy nodes are updated
    based on their k-nearest neighbors.

    Args:
        xyz: [N, D] numpy array of spatial coordinates (only [:, :3] used)
        probs: [N, C] numpy array of class probabilities per node
        k: number of nearest neighbors
        entropy_threshold: nodes with entropy below this are considered fixed
        n_iter: number of propagation iterations
        seed: optional, for reproducibility

    Returns:
        labels: final predicted labels
        fixed_mask: boolean mask indicating which nodes were fixed
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if seed is not None:
        np.random.seed(seed)

    xyz = np.asarray(xyz)
    probs = np.asarray(probs)
    N, C = probs.shape

    # Compute entropy for each node
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    fixed_mask = entropy < entropy_threshold

    # Initialize labels with argmax
    labels = np.argmax(probs, axis=1)

    # Build graph
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(xyz[:, :3])
    _, indices = nbrs.kneighbors(xyz[:, :3])
    neighbor_indices = indices[:, 1:]  # skip self

    for _ in range(n_iter):
        new_labels = labels.copy()
        for i in range(N):
            if fixed_mask[i]:
                continue
            neighbor_labels = labels[neighbor_indices[i]]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            new_labels[i] = values[np.argmax(counts)]
        labels = new_labels

    return labels, fixed_mask

def gibbs_sample_ugm(xyz, marginals, k=5, coupling=1.0, n_iter=10, seed=None):
    """
    Gibbs sampling from a pairwise UGM with model marginals as unary potentials
    and Ising-style pairwise coupling encouraging label agreement.

    Args:
        xyz: [N, D] spatial coordinates (only [:, :3] used)
        marginals: [N, C] model-predicted class probabilities (unary potentials)
        k: number of nearest neighbors
        coupling: float, controls how strongly neighbors are encouraged to agree
        n_iter: number of Gibbs sweeps
        seed: random seed

    Returns:
        labels: [N] sampled label configuration
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if seed is not None:
        np.random.seed(seed)

    xyz = np.asarray(xyz)
    marginals = np.asarray(marginals)
    N, C = marginals.shape

    # Build neighbor graph
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(xyz[:, :3])
    _, indices = nbrs.kneighbors(xyz[:, :3])
    neighbor_indices = indices[:, 1:]

    # Initialize labels from marginals
    labels = np.array([np.random.choice(C, p=marginals[i]) for i in range(N)])

    for _ in range(n_iter):
        for i in range(N):
            unary_logp = np.log(marginals[i] + 1e-12)  # [C]

            # Pairwise term: encourage same label as neighbors
            neighbors = neighbor_indices[i]
            neighbor_labels = labels[neighbors]

            counts = np.bincount(neighbor_labels, minlength=C)  # [C]
            pairwise_logp = coupling * counts

            logp = unary_logp + pairwise_logp
            logp -= np.max(logp)  # For numerical stability
            p = np.exp(logp)
            p /= p.sum()

            labels[i] = np.random.choice(C, p=p)

    return labels


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



def generate_anndata_from_samples_with_budget(region_model, xyz, device="cuda", sample_from_probs=False):
    counts=[]
    for token in range(slice_data_loader.gene_exp_model.num_tokens):
        counts.append((slice_data_loader.test_slices[0].obs["token"]==token).sum())

    budget=np.array(counts)    

    region_model.eval().to(device)

    with torch.no_grad():
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).to(device)
        probs = torch.softmax(region_model.model(xyz_tensor), dim=1).cpu().numpy()
        preds=[]
        pred_tree = cKDTree(xyz)

        for i,p in enumerate(probs):
            new_prob_vec= ((len(probs)-i)*p + 0.001*i*(budget/budget.sum()))/((len(probs)-i)*p + 0.001*i*(budget/budget.sum())).sum()
            gt_distances, gt_indices =pred_tree.query(xyz[i], k=20)
            chosen_token=np.random.choice(len(p), p=new_prob_vec)
            if budget[chosen_token]>0:
                budget[chosen_token]=budget[chosen_token]-1
            preds.append(chosen_token)
        # preds = np.array([np.random.choice(len(p), p=p) for p in probs]) if sample_from_probs else np.argmax(probs, axis=1)
        preds=np.array(preds)
    adata = ad.AnnData(X=np.zeros((xyz.shape[0], 1)))
    adata.obsm["spatial"] = xyz[:, :3].cpu().numpy()
    adata.obs["token"] = preds
    return adata, preds



def generate_baseline1_majority(test_locations, reference_locations, reference_celltypes, k=20):
    """
    For each point in test_locations, find the k nearest neighbors in reference_locations
    and return the majority vote among their reference_celltypes.

    Args:
        test_locations: array-like of shape (N_test, D)
        reference_locations: array-like of shape (N_ref, D)
        reference_celltypes: array-like of shape (N_ref,)
        k: number of nearest neighbors to consider

    Returns:
        preds: numpy array of shape (N_test,), predicted celltype for each test point
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Helper: convert to numpy if not already
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        try:
            return x.cpu().numpy()
        except:
            return np.asarray(x)

    test_np = to_numpy(test_locations)
    ref_np = to_numpy(reference_locations)
    labels_np = to_numpy(reference_celltypes).astype(int)

    # Build kNN index on reference locations
    nbrs = NearestNeighbors(n_neighbors=k).fit(ref_np)
    distances, indices = nbrs.kneighbors(test_np)  # indices shape = (N_test, k)

    n_test = test_np.shape[0]
    preds = np.zeros(n_test, dtype=int)

    for i in range(n_test):
        neighbor_idxs = indices[i]                          # shape (k,)
        neighbor_labels = labels_np[neighbor_idxs]          # shape (k,)
        # Compute majority vote (mode)
        counts = np.bincount(neighbor_labels)
        preds[i] = np.argmax(counts)

    return preds
