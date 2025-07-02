# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def one_hot_encode(celltypes):
#     unique_types = list(set(celltypes))
#     encoding_dict = {ctype: np.eye(len(unique_types))[i] for i, ctype in enumerate(unique_types)}
#     return encoding_dict, len(unique_types)

# def get_neighbors(gt_positions, target_pos, radius):
#     return [i for i, pos in enumerate(gt_positions) if np.linalg.norm(np.array(pos) - np.array(target_pos)) <= radius]

# def soft_accuracy(gt_celltypes, gt_positions, pred_celltypes, pred_positions, radius):
#     encoding_dict, num_classes = one_hot_encode(gt_celltypes + pred_celltypes)
#     result = []
    
#     for i, pos in enumerate(gt_positions):
#         gt_neighbors = get_neighbors(gt_positions, pos, radius)
#         pred_neighbors = get_neighbors(pred_positions, pos, radius)
        
#         gt_encoding_sum = np.sum([encoding_dict[gt_celltypes[j]] for j in gt_neighbors], axis=0)
#         pred_encoding_sum = np.sum([encoding_dict[pred_celltypes[j]] for j in pred_neighbors], axis=0)
        
#         gt_distribution = gt_encoding_sum / np.linalg.norm(gt_encoding_sum) if np.linalg.norm(gt_encoding_sum) != 0 else np.zeros(num_classes)
#         pred_distribution = pred_encoding_sum / np.linalg.norm(pred_encoding_sum) if np.linalg.norm(pred_encoding_sum) != 0 else np.zeros(num_classes)
        
#         similarity = cosine_similarity(gt_distribution.reshape(1, -1), pred_distribution.reshape(1, -1))[0, 0]
#         result.append(similarity)
    
#     return np.mean(result) if result else 0.0


import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
import tqdm
def one_hot_encode(celltypes):
    unique_types = list(set(celltypes))
    encoding_dict = {ctype: np.eye(len(unique_types))[i] for i, ctype in enumerate(unique_types)}
    return encoding_dict, len(unique_types)

def soft_accuracy(gt_celltypes, gt_positions, pred_celltypes, pred_positions, radius=None, k=0, return_list=False, return_percent=False, sample=None):
    encoding_dict, num_classes = one_hot_encode(gt_celltypes + pred_celltypes)

    counts=np.sum([encoding_dict[ct] for ct in gt_celltypes],axis=0)/np.sum([encoding_dict[ct] for ct in gt_celltypes])


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


    for i, pos in tqdm.tqdm(list(enumerate(samples))):
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
        return [(gt_distribution*counts).sum() for gt_distribution in gt_distributions]
    if return_list:
        return result

    return np.mean(result) if result else 0.0



def diversity(gt_celltypes, gt_positions, radius):
    encoding_dict, num_classes = one_hot_encode(gt_celltypes)

    gt_positions = np.array(gt_positions)

    gt_tree = cKDTree(gt_positions)
    result = []
    for i, pos in tqdm.tqdm(list(enumerate(gt_positions))):
        gt_neighbors = gt_tree.query_ball_point(pos, radius)

        gt_encoding_sum = np.sum([encoding_dict[gt_celltypes[j]] for j in gt_neighbors], axis=0)
        result.append((gt_encoding_sum>0).sum())
    return np.array(result)




import numpy as np
import tqdm
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

def soft_correlation(gt_expressions, gt_positions, pred_expressions, pred_positions, radius=None, k=0, sample=None):
    """
    gt_expressions, pred_expressions: list or array of gene expression vectors (shape [num_cells, num_genes])
    gt_positions, pred_positions: list or array of positions (shape [num_cells, 2] or [num_cells, 3])
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """

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

    for i, pos in tqdm.tqdm(list(enumerate(samples))):
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

        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} samples...")

    gt_sums = np.array(gt_sums).flatten()
    pred_sums = np.array(pred_sums).flatten()

    if len(gt_sums) == 0 or len(pred_sums) == 0:
        return 0.0

    correlation, _ = pearsonr(gt_sums, pred_sums)
    return correlation






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
