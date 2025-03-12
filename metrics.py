import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def one_hot_encode(celltypes):
    unique_types = list(set(celltypes))
    encoding_dict = {ctype: np.eye(len(unique_types))[i] for i, ctype in enumerate(unique_types)}
    return encoding_dict, len(unique_types)

def get_neighbors(gt_positions, target_pos, radius):
    return [i for i, pos in enumerate(gt_positions) if np.linalg.norm(np.array(pos) - np.array(target_pos)) <= radius]

def soft_accuracy(gt_celltypes, gt_positions, pred_celltypes, pred_positions, radius):
    encoding_dict, num_classes = one_hot_encode(gt_celltypes + pred_celltypes)
    result = []
    
    for i, pos in enumerate(gt_positions):
        gt_neighbors = get_neighbors(gt_positions, pos, radius)
        pred_neighbors = get_neighbors(pred_positions, pos, radius)
        
        gt_encoding_sum = np.sum([encoding_dict[gt_celltypes[j]] for j in gt_neighbors], axis=0)
        pred_encoding_sum = np.sum([encoding_dict[pred_celltypes[j]] for j in pred_neighbors], axis=0)
        
        gt_distribution = gt_encoding_sum / np.linalg.norm(gt_encoding_sum) if np.linalg.norm(gt_encoding_sum) != 0 else np.zeros(num_classes)
        pred_distribution = pred_encoding_sum / np.linalg.norm(pred_encoding_sum) if np.linalg.norm(pred_encoding_sum) != 0 else np.zeros(num_classes)
        
        similarity = cosine_similarity(gt_distribution.reshape(1, -1), pred_distribution.reshape(1, -1))[0, 0]
        result.append(similarity)
    
    return np.mean(result) if result else 0.0
