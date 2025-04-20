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


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
import tqdm
def one_hot_encode(celltypes):
    unique_types = list(set(celltypes))
    encoding_dict = {ctype: np.eye(len(unique_types))[i] for i, ctype in enumerate(unique_types)}
    return encoding_dict, len(unique_types)

def soft_accuracy(gt_celltypes, gt_positions, pred_celltypes, pred_positions, radius, return_list=False, return_percent=False):
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
    for i, pos in tqdm.tqdm(list(enumerate(gt_positions))):
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
