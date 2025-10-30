import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
import tqdm
from scipy.spatial import Delaunay
import numpy as np
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import pandas as pd

def one_hot_encode(celltypes):
    unique_types = list(set(celltypes))
    I = np.eye(len(unique_types), dtype=np.float32)
    encoding_dict = {ctype: I[i] for i, ctype in enumerate(unique_types)}
    return encoding_dict, len(unique_types)

def soft_accuracy(gt_celltypes, gt_positions, pred_celltypes, pred_positions, radius=None, k=0, return_list=False, return_percent=False, sample=None):
    encoding_dict, num_classes = one_hot_encode(gt_celltypes + pred_celltypes)



    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    result = []
    gt_distributions = []
    pred_distributions = []
    gt_neighbors_all=[]
    pred_neighbors_all=[]

    if sample is not None:
        percent = sample
        n = int(len(gt_positions) * percent / 100)
        indices = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[indices]

    else:
        samples=gt_positions


    for i, pos in tqdm(list(enumerate(samples))):
        if k>0:
            gt_distances, gt_indices = gt_tree.query(pos, k=k+1)
            pred_distances, pred_indices = pred_tree.query(pos, k=k+1)
            gt_neighbors = gt_indices[1:]
            pred_neighbors = pred_indices[1:]            
        else:
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)
        gt_neighbors_all.append(gt_neighbors)
        pred_neighbors_all.append(pred_neighbors)

        gt_encoding_sum = np.sum([encoding_dict[gt_celltypes[j]] for j in gt_neighbors], axis=0)
        pred_encoding_sum = np.sum([encoding_dict[pred_celltypes[j]] for j in pred_neighbors], axis=0)

        # Normalize
        gt_norm = np.linalg.norm(gt_encoding_sum)
        pred_norm = np.linalg.norm(pred_encoding_sum)

        gt_distribution = gt_encoding_sum / gt_norm if gt_norm != 0 else np.zeros(num_classes)
        pred_distribution = pred_encoding_sum / pred_norm if pred_norm != 0 else np.zeros(num_classes)
        gt_distributions.append(gt_distribution)
        pred_distributions.append(pred_distribution)

        similarity = cosine_similarity(gt_distribution.reshape(1, -1), pred_distribution.reshape(1, -1))[0, 0]
        result.append(similarity)

        if i%10000==0:
            print(np.mean(result))

    if return_percent:
        counts=np.sum([encoding_dict[ct] for ct in gt_celltypes],axis=0)/np.sum([encoding_dict[ct] for ct in gt_celltypes])
        return [(gt_distribution*counts).sum() for gt_distribution in gt_distributions]
    if return_list:
        return result

    return np.mean(result) if result else 0.0


from scipy.spatial import Delaunay
import numpy as np

import numpy as np
from collections import Counter
from scipy.spatial import Delaunay

def delauney_colocalization(gt_celltypes, gt_positions, pred_celltypes, pred_positions, encoding_dict=None):
    """
    Build Delaunay graph for GT and Pred positions and compute L1 distance
    between their edge-type count maps (sparse via Counter).

    Parameters
    ----------
    gt_celltypes : list[str]
        Ground-truth cell type labels (length N).
    gt_positions : array-like, shape (N, 2)
        Ground-truth positions.
    pred_celltypes : list[str]
        Predicted cell type labels (length M).
    pred_positions : array-like, shape (M, 2)
        Predicted positions.
    encoding_dict : dict, optional
        Mapping from cell type to index. If None, builds internally.

    Returns
    -------
    l1_distance : float
        L1 distance (sum of absolute differences) between GT and Pred edge-type counts.
    """

    # Build encoding dict if not given
    if encoding_dict is None:
        all_cts = list(set(gt_celltypes) | set(pred_celltypes))
        encoding_dict = {ct: i for i, ct in enumerate(all_cts)}

    def build_count_counter(celltypes, positions):
        tri = Delaunay(positions)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = sorted((simplex[i], simplex[(i+1)%3]))
                edges.add((a, b))

        counter = Counter()
        for a, b in edges:
            ia, ib = encoding_dict[celltypes[a]], encoding_dict[celltypes[b]]
            counter[(ia, ib)] += 1
            if ia != ib:
                counter[(ib, ia)] += 1  # keep symmetric
        return counter

    gt_counter = build_count_counter(gt_celltypes, np.array(gt_positions))
    pred_counter = build_count_counter(pred_celltypes, np.array(pred_positions))

    all_keys = set(gt_counter.keys()) | set(pred_counter.keys())
    l1_distance = sum(abs(gt_counter[k] - pred_counter[k]) for k in all_keys)

    return l1_distance / len(gt_positions)




def gridized_l1_distance(gt_positions, pred_positions, radius=None, k=0, 
                              grid_size=50, return_list=False):
    """
    Estimate weighted L1 distance between two distributions by evaluating
    density differences on a regular grid of points spanning the data space.
    
    Parameters
    ----------
    gt_positions : array-like, shape (n_gt, d)
        Ground-truth positions.
    pred_positions : array-like, shape (n_pred, d)
        Predicted positions.
    radius : float, optional
        Radius for density estimation (used if k == 0).
    k : int, optional
        k for kNN density estimation. If > 0, kNN mode is used.
    grid_size : int, optional
        Number of grid points per dimension.
    return_list : bool, optional
        If True, return list of per-grid-point distances. Otherwise return mean.
    """
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    d = gt_positions.shape[1]
    Vd = np.pi ** (d/2) / np.math.gamma(d/2 + 1)  # volume of unit d-ball
    n_gt, n_pred = len(gt_positions), len(pred_positions)

    # Build trees
    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    # Grid bounding box
    mins = np.minimum(gt_positions.min(axis=0), pred_positions.min(axis=0))
    maxs = np.maximum(gt_positions.max(axis=0), pred_positions.max(axis=0))
    grid_axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(d)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=-1)

    results = []

    for i, pos in tqdm(list(enumerate(grid_points))):
        if k > 0:
            # kNN mode
            gt_distances, _ = gt_tree.query(pos, k=k)
            pred_distances, _ = pred_tree.query(pos, k=k)
            r_gt = gt_distances[-1]
            r_pred = pred_distances[-1]

            p_hat = k / (n_gt * Vd * (r_gt ** d)) if r_gt > 0 else 0
            q_hat = k / (n_pred * Vd * (r_pred ** d)) if r_pred > 0 else 0
        else:
            # Radius mode
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)

            k_gt = len(gt_neighbors)
            k_pred = len(pred_neighbors)

            p_hat = k_gt / (n_gt * Vd * (radius ** d)) if radius > 0 else 0
            q_hat = k_pred / (n_pred * Vd * (radius ** d)) if radius > 0 else 0

        results.append(abs(p_hat - q_hat))

    return results if return_list else np.mean(results) if results else 0.0


import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

def gridized_kl_divergence(gt_positions, pred_positions, radius=None, k=0, 
                           grid_size=50, return_list=False, eps=1e-200):
    """
    Estimate KL divergence KL(P||Q) between two distributions by evaluating
    density estimates on a regular grid of points spanning the data space.

    Parameters
    ----------
    gt_positions : array-like, shape (n_gt, d)
        Ground-truth samples (distribution P).
    pred_positions : array-like, shape (n_pred, d)
        Predicted samples (distribution Q).
    radius : float, optional
        Radius for density estimation (used if k == 0).
    k : int, optional
        k for kNN density estimation. If > 0, kNN mode is used.
    grid_size : int, optional
        Number of grid points per dimension.
    return_list : bool, optional
        If True, return list of per-grid-point KL terms. Otherwise return mean.
    eps : float, optional
        Small constant to avoid log(0) and division by zero.
    """
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    d = gt_positions.shape[1]
    Vd = np.pi ** (d/2) / np.math.gamma(d/2 + 1)  # volume of unit d-ball
    n_gt, n_pred = len(gt_positions), len(pred_positions)

    # Build trees
    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    # Grid bounding box
    mins = np.minimum(gt_positions.min(axis=0), pred_positions.min(axis=0))
    maxs = np.maximum(gt_positions.max(axis=0), pred_positions.max(axis=0))
    grid_axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(d)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=-1)

    results = []

    for i, pos in tqdm(list(enumerate(grid_points))):
        if k > 0:
            # kNN mode
            gt_distances, _ = gt_tree.query(pos, k=k)
            pred_distances, _ = pred_tree.query(pos, k=k)
            r_gt = gt_distances[-1]
            r_pred = pred_distances[-1]

            p_hat = k / (n_gt * Vd * (r_gt ** d)) if r_gt > 0 else 0.0
            q_hat = k / (n_pred * Vd * (r_pred ** d)) if r_pred > 0 else 0.0
        else:
            # Radius mode
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)

            k_gt = len(gt_neighbors)
            k_pred = len(pred_neighbors)

            p_hat = k_gt / (n_gt * Vd * (radius ** d)) if radius > 0 else 0.0
            q_hat = k_pred / (n_pred * Vd * (radius ** d)) if radius > 0 else 0.0

        # Add eps to avoid instability
        p_hat = max(p_hat, eps)
        q_hat = max(q_hat, eps)

        results.append(p_hat * np.log(p_hat / q_hat))

    return results if return_list else np.mean(results) if results else 0.0

# def weighted_l1_distance(gt_positions, pred_positions, radius=None, k=0, 
#                          return_list=False, sample=None):
#     """
#     Estimate weighted L1 distance between two distributions
#     represented by samples (gt_positions vs pred_positions).
    
#     Uses kNN or radius-based neighborhood density estimates.
#     """
#     gt_positions = np.array(gt_positions)
#     pred_positions = np.array(pred_positions)

#     # Build trees for neighbor lookup
#     gt_tree = cKDTree(gt_positions)
#     pred_tree = cKDTree(pred_positions)

#     # Optionally subsample GT samples
#     if sample is not None:
#         percent = sample
#         n = int(len(gt_positions) * percent / 100)
#         indices = np.random.choice(len(gt_positions), size=n, replace=False)
#         samples = gt_positions[indices]
#     else:
#         samples = gt_positions

#     d = gt_positions.shape[1]  # dimension
#     Vd = np.pi ** (d/2) / np.math.gamma(d/2 + 1)  # volume of unit d-ball
#     n_gt, n_pred = len(gt_positions), len(pred_positions)

#     results = []

#     for i, pos in tqdm(list(enumerate(samples))):
#         if k > 0:
#             # kNN mode: get kth neighbor distance
#             gt_distances, _ = gt_tree.query(pos, k=k+1)  # include self
#             pred_distances, _ = pred_tree.query(pos, k=k)
#             r_gt = gt_distances[-1]  # k-th neighbor
#             r_pred = pred_distances[-1] if len(pred_distances) > 0 else np.inf

#             p_hat = k / (n_gt * Vd * (r_gt ** d)) if r_gt > 0 else 0
#             q_hat = k / (n_pred * Vd * (r_pred ** d)) if r_pred > 0 else 0

#         else:
#             # Radius mode: count neighbors within radius
#             gt_neighbors = gt_tree.query_ball_point(pos, radius)
#             pred_neighbors = pred_tree.query_ball_point(pos, radius)

#             k_gt = max(len(gt_neighbors) - 1, 0)  # exclude self
#             k_pred = len(pred_neighbors)

#             p_hat = k_gt / (n_gt * Vd * (radius ** d)) if radius > 0 else 0
#             q_hat = k_pred / (n_pred * Vd * (radius ** d)) if radius > 0 else 0

#         results.append(abs(p_hat - q_hat))

#         if i % 10000 == 0 and i > 0:
#             print("Running mean:", np.mean(results))

#     return results if return_list else np.mean(results) if results else 0.0



def diversity(gt_celltypes, gt_positions, radius):
    encoding_dict, num_classes = one_hot_encode(gt_celltypes)

    gt_positions = np.array(gt_positions)

    gt_tree = cKDTree(gt_positions)
    result = []
    for i, pos in tqdm(list(enumerate(gt_positions))):
        gt_neighbors = gt_tree.query_ball_point(pos, radius)

        gt_encoding_sum = np.sum([encoding_dict[gt_celltypes[j]] for j in gt_neighbors], axis=0)
        result.append((gt_encoding_sum>0).sum())
    return np.array(result)


import numpy as np
import scipy.sparse as sp

def _nnz_per_gene(X):
    if sp.issparse(X):
        return np.asarray(X.getnnz(axis=0)).ravel()
    return np.asarray((X > 0).sum(axis=0)).ravel()

def intersect_and_filter_X(gt_adata, pred_adata, min_expr_cells=0, gene_set=None):
    # 1) intersect genes (order is preserved by AnnData slicing)
    common_genes = gt_adata.var_names.intersection(pred_adata.var_names)
    if gene_set is not None:
        common_genes = common_genes.intersection(np.asarray(gene_set))


    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between gt_adata and pred_adata.")
    print(gt_adata.var_names)
    gt_common   = gt_adata[:, common_genes]
    pred_common = pred_adata[:, common_genes]
    print(len(common_genes))

    # 2) require expression in each adata
    gt_nnz   = _nnz_per_gene(gt_common.X)
    pred_nnz = _nnz_per_gene(pred_common.X)
    keep_mask = (gt_nnz >= min_expr_cells) & (pred_nnz >= min_expr_cells)

    if not np.any(keep_mask):
        raise ValueError("No genes pass the expression filter in both adatas.")

    # 3) return filtered .X matrices and the kept gene names (same order)
    gt_X_filtered   = gt_common[:, keep_mask].X
    pred_X_filtered = pred_common[:, keep_mask].X
    kept_genes = common_genes[keep_mask]
    if sp.issparse(gt_X_filtered):
        gt_X_filtered = gt_X_filtered.todense()
    if sp.issparse(pred_X_filtered):
        pred_X_filtered = pred_X_filtered.todense()

    return gt_X_filtered.tolist(), pred_X_filtered.tolist(), kept_genes




import numpy as np
import tqdm
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

def soft_correlation(gt_adata, gt_positions, pred_adata, pred_positions, radius=None, k=0, sample=None, return_list=False, corr_type='pearson', gene_set=None):
    """
    gt_expressions, pred_expressions: list or array of gene expression vectors (shape [num_cells, num_genes])
    gt_positions, pred_positions: list or array of positions (shape [num_cells, 2] or [num_cells, 3])
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """

    if corr_type == "pearson":          
        corr_fn = pearsonr                      
    elif corr_type == "spearman":                 
        corr_fn = spearmanr   
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(gt_adata, pred_adata, 1, gene_set)
    print(f'running soft_correlation on {len(genes)} genes')
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)
    gt_expressions = np.array(gt_expressions)
    pred_expressions = np.array(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    gt_sums = []
    pred_sums = []

    if sample is not None:
        percent = sample
        n = int(len(gt_positions) * percent / 100)
        indices = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[indices]
    else:
        samples = gt_positions

    correlations_all = []
    for i, pos in tqdm(list(enumerate(samples))):
        if k > 0:
            gt_distances, gt_indices = gt_tree.query(pos, k=k+1)
            pred_distances, pred_indices = pred_tree.query(pos, k=k+1)
            gt_neighbors = gt_indices[1:]  # exclude self
            pred_neighbors = pred_indices[1:]
        else:
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)

        gt_sum = np.sum(gt_expressions[gt_neighbors], axis=0)
        pred_sum = np.sum(pred_expressions[pred_neighbors], axis=0)

        gt_sums.append(gt_sum)
        pred_sums.append(pred_sum)

        pred_sum[0]=pred_sum[0]+1e-15  # to avoid NaN in pearsonr when pred_sum is all zeros

        correlations_all.append(corr_fn(gt_sum, pred_sum)[0])
            
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} samples...")

    gt_sums = np.array(gt_sums).flatten()
    pred_sums = np.array(pred_sums).flatten()

    if len(gt_sums) == 0 or len(pred_sums) == 0:
        return 0.0

    if return_list:
        return correlations_all

    correlation, _ = corr_fn(gt_sums, pred_sums)
    return correlation




def soft_gene_distance(gt_adata, gt_positions, pred_adata, pred_positions, radius=None, k=0, sample=None):
    """
    gt_expressions, pred_expressions: list or array of gene expression vectors (shape [num_cells, num_genes])
    gt_positions, pred_positions: list or array of positions (shape [num_cells, 2] or [num_cells, 3])
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(gt_adata, pred_adata, 0)
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)
    gt_expressions = np.array(gt_expressions)
    pred_expressions = np.array(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    gt_sums = []
    pred_sums = []
    dists_all = []

    if sample is not None:
        percent = sample
        n = int(len(gt_positions) * percent / 100)
        indices = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[indices]
    else:
        samples = gt_positions

    for i, pos in tqdm(list(enumerate(samples))):
        if k > 0:
            gt_distances, gt_indices = gt_tree.query(pos, k=k+1)
            pred_distances, pred_indices = pred_tree.query(pos, k=k+1)
            gt_neighbors = gt_indices[1:]  # exclude self
            pred_neighbors = pred_indices[1:]
        else:
            gt_neighbors = gt_tree.query_ball_point(pos, radius)
            pred_neighbors = pred_tree.query_ball_point(pos, radius)

        gt_sum = np.sum(gt_expressions[gt_neighbors], axis=0)
        pred_sum = np.sum(pred_expressions[pred_neighbors], axis=0)

        dists_all.append(np.linalg.norm(gt_sum - pred_sum))

        gt_sums.append(gt_sum)
        pred_sums.append(pred_sum)

        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} samples...")

    gt_sums = np.array(gt_sums).flatten()
    pred_sums = np.array(pred_sums).flatten()

    if len(gt_sums) == 0 or len(pred_sums) == 0:
        return 0.0

    # correlation, _ = pearsonr(gt_sums, pred_sums)
    return float(np.mean(dists_all)) if dists_all else 0.0



import numpy as np
from scipy.spatial import cKDTree
import tqdm

def soft_precision(gt_adata, gt_positions, pred_adata, pred_positions, radius=None, k=0, sample=None):
    """
    gt_expressions, pred_expressions: array of shape [num_cells, num_genes]
    gt_positions, pred_positions: array of shape [num_cells, 2] or [num_cells, 3]
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(gt_adata, pred_adata)
    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)
    gt_expressions = np.asarray(gt_expressions)
    pred_expressions = np.asarray(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    precisions = []

    # sampling
    if sample is not None:
        n = int(len(gt_positions) * sample / 100)
        idx = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[idx]
    else:
        samples = gt_positions

    for i, pos in enumerate(tqdm(samples, desc="spots")):
        # find neighbors
        if k > 0:
            _, gt_idx = gt_tree.query(pos, k=k+1)
            _, pred_idx = pred_tree.query(pos, k=k+1)
            gt_nbrs = gt_idx[1:]
            pred_nbrs = pred_idx[1:]
        else:
            gt_nbrs = gt_tree.query_ball_point(pos, radius)
            pred_nbrs = pred_tree.query_ball_point(pos, radius)

        # sum expressions over neighbors
        gt_sum = np.sum(gt_expressions[gt_nbrs], axis=0)
        pred_sum = np.sum(pred_expressions[pred_nbrs], axis=0)

        # compute precision: TP / (predicted positives)
        pred_pos = pred_sum > 0
        if pred_pos.sum() > 0:
            true_pos = np.logical_and(pred_pos, gt_sum > 0).sum()
            precisions.append(true_pos / pred_pos.sum())
        else:
            precisions.append(0.0)

        if i and i % 10000 == 0:
            print(f"Processed {i} spots...")

    return float(np.mean(precisions)) if precisions else 0.0




def soft_f1(gt_adata, gt_positions, pred_adata, pred_positions, radius=None, k=0, sample=None, return_list=False):
    """
    gt_expressions, pred_expressions: array of shape [num_cells, num_genes]
    gt_positions, pred_positions: array of shape [num_cells, 2] or [num_cells, 3]
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(gt_adata, pred_adata)
    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)
    gt_expressions = np.asarray(gt_expressions)
    pred_expressions = np.asarray(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    precisions = []
    recalls = []
    f1s = []

    # sampling
    if sample is not None:
        n = int(len(gt_positions) * sample / 100)
        idx = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[idx]
    else:
        samples = gt_positions

    for i, pos in enumerate(tqdm(samples, desc="spots")):
        # find neighbors
        if k > 0:
            _, gt_idx = gt_tree.query(pos, k=k+1)
            _, pred_idx = pred_tree.query(pos, k=k+1)
            gt_nbrs = gt_idx[1:]
            pred_nbrs = pred_idx[1:]
        else:
            gt_nbrs = gt_tree.query_ball_point(pos, radius)
            pred_nbrs = pred_tree.query_ball_point(pos, radius)

        # sum expressions over neighbors
        gt_sum = np.sum(gt_expressions[gt_nbrs], axis=0)
        pred_sum = np.sum(pred_expressions[pred_nbrs], axis=0)

        # compute precision: TP / (predicted positives)
        pred_pos = pred_sum > 0
        if pred_pos.sum() > 0:
            true_pos = np.logical_and(pred_pos, gt_sum > 0).sum()
            precisions.append(true_pos / pred_pos.sum())
            recalls.append(true_pos / (gt_sum > 0).sum() if (gt_sum > 0).sum() > 0 else 0.0)
            f1s.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1]) if (precisions[-1] + recalls[-1]) > 0 else 0.0)
        else:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)

        if i and i % 10000 == 0:
            print(f"Processed {i} spots...")

    if return_list:
        return f1s
    return float(np.mean(f1s)) if f1s else 0.0, float(np.mean(precisions)) if precisions else 0.0, float(np.mean(recalls)) if recalls else 0.0






def lookup_correlation(gt_celltypes, gt_positions, gt_expressions,
                       pred_celltypes, pred_positions, radius=None, k=0, sample=None):
    """
    For each predicted cell, find the nearest ground-truth cell with the same celltype
    and use its gene expression as the prediction. Then compute local correlation.

    Arguments:
    - gt_celltypes, pred_celltypes: list of cell type labels
    - gt_positions, pred_positions: list/array of [x,y] coordinates
    - gt_expressions: [num_gt_cells x num_genes] array of ground-truth gene expressions
    - radius / k / sample: same as in soft_correlation
    """

    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)
    gt_expressions = np.array(gt_expressions)

    pred_expressions_lookup = np.zeros((len(pred_positions), gt_expressions.shape[1]))

    gt_by_type = {}
    for i, ct in enumerate(gt_celltypes):
        gt_by_type.setdefault(ct, []).append(i)

    for i, (ct, pos) in enumerate(zip(pred_celltypes, pred_positions)):
        if ct not in gt_by_type:
            continue
        indices = gt_by_type[ct]
        gt_subset = gt_positions[indices]
        tree = cKDTree(gt_subset)
        _, idx = tree.query(pos, k=1)
        pred_expressions_lookup[i] = gt_expressions[indices[idx]]

    # Now use soft_correlation to compare aggregated values
    return soft_correlation(
        gt_expressions=gt_expressions,
        gt_positions=gt_positions,
        pred_expressions=pred_expressions_lookup,
        pred_positions=pred_positions,
        radius=radius,
        k=k,
        sample=sample
    )










import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from tqdm import tqdm

# def one_hot_encode(labels):
#     """Map each label to an integer class and build one‑hot matrix."""
#     classes = sorted(set(labels))
#     idx = {c:i for i,c in enumerate(classes)}
#     num = len(classes)
#     return idx, num

def _compute_enrichment_z(labels, positions, radius=None, k=0, n_permutations=1000, seed=None, C=0):
    """
    Compute the neighborhood enrichment z‑score matrix for one dataset.
    
    labels : list of hashable
        Cell/spot cluster assignments.
    positions : array (N,2 or 3)
        x/y (or x/y/z) coordinates.
    radius : float, optional
        If >0, use radius‐based neighbors.
    k : int, optional
        If >0, use k nearest neighbors.
    n_permutations : int
        Number of random label shuffles to build null distribution.
    seed : int, optional
        RNG seed for reproducibility.
        
    Returns
    -------
    z : array (C,C)
        z[i,j] is the enrichment of cluster i next to cluster j.
    """
    if (radius is None and k == 0) or (radius is not None and k > 0):
        raise ValueError("Specify exactly one of `radius` or `k>0`.")
    
    labels = list(labels)
    positions = np.asarray(positions)
    # idx_map, C = one_hot_encode(labels)
    # C = max(labels)+1

    L = np.array(labels, dtype=int)
    N = len(L)
    
    # build neighbor lists once
    tree = cKDTree(positions)
    neighbors = []
    if k > 0:
        dists, inds = tree.query(positions, k=k+1)
        for u, nbrs in enumerate(inds):
            # drop self
            neighbors.append([v for v in nbrs if v != u])
    else:
        for u, pos in enumerate(positions):
            nbrs = tree.query_ball_point(pos, radius)
            neighbors.append([v for v in nbrs if v != u])
    
    # observed counts
    obs = np.zeros((C, C), dtype=int)
    for u in range(N):
        i = L[u]
        for v in neighbors[u]:
            obs[i, L[v]] += 1
    return obs
    
    # # null distributions
    # rng = np.random.default_rng(seed)
    # perm_counts = np.zeros((n_permutations, C, C), dtype=float)
    # for t in tqdm(range(n_permutations), desc="permutations", leave=False):
    #     perm = rng.permutation(L)
    #     cnt = np.zeros((C, C), dtype=int)
    #     for u in range(N):
    #         i = perm[u]
    #         for v in neighbors[u]:
    #             cnt[i, perm[v]] += 1
    #     perm_counts[t] = cnt
    
    # mu = perm_counts.mean(axis=0)
    # sigma = perm_counts.std(axis=0, ddof=1)
    # # avoid division by zero
    # sigma[sigma == 0] = 1.0
    # z = (obs - mu) / sigma
    # return z

import numpy as np

def _realign_z(z, original_classes, target_classes, pad_value=0.0):
    """
    Take a (C0×C0) matrix `z` whose rows/cols correspond to original_classes,
    and produce a (C×C) matrix in the order of target_classes, filling missing
    entries with pad_value.
    """
    C = len(target_classes)
    idx0 = {c:i for i, c in enumerate(original_classes)}
    Z = np.full((C, C), pad_value, dtype=z.dtype)
    for i, ci in enumerate(target_classes):
        if ci not in idx0:
            continue
        ii = idx0[ci]
        for j, cj in enumerate(target_classes):
            if cj not in idx0:
                continue
            jj = idx0[cj]
            Z[i, j] = z[ii, jj]
    return Z

def neighborhood_enrichment(
    gt_celltypes, gt_positions,
    pred_celltypes, pred_positions,
    radius=None, k=0, n_permutations=1000,
    return_matrix=False, seed=None
):
    """
    Compare spatial neighborhood enrichment between ground truth and prediction.
    
    gt_celltypes, pred_celltypes : list of labels
    gt_positions,   pred_positions   : list or array of coordinates
    radius : float
        Neighborhood radius.
    k : int
        Number of nearest neighbors.
    n_permutations : int
        How many random shuffles to build null.
    return_matrix : bool
        If True, return (z_gt, z_pred) matrices instead of a single score.
    return_list : bool
        If True, return a list of ((i,j), z_gt, z_pred) for all cluster pairs.
    seed : int
        RNG seed.
        
    Returns
    -------
    If neither return_matrix nor return_list:
        float
            Pearson correlation between flattened z‑score matrices.
    If return_matrix:
        z_gt, z_pred : arrays (C,C)
    If return_list:
        list of ((cluster_i, cluster_j), z_gt[i,j], z_pred[i,j])
    """
    z_gt   = _compute_enrichment_z(gt_celltypes,   gt_positions,   radius, k, n_permutations, seed, C=max(max(gt_celltypes),max(pred_celltypes))+1)
    z_pred = _compute_enrichment_z(pred_celltypes, pred_positions, radius, k, n_permutations, seed, C=max(max(gt_celltypes),max(pred_celltypes))+1)
    # ensure same shape    

    classes = sorted(set(gt_celltypes) | set(pred_celltypes))



    if return_matrix:
        return z_gt, z_pred


    flat_gt   = z_gt.ravel()
    flat_pred = z_pred.ravel()
    
    
    # default: one scalar summary
    corr, _ = pearsonr(flat_gt, flat_pred)
    return corr





