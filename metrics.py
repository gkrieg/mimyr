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

def intersect_and_filter_X(gt_adata, pred_adata, min_expr_cells=0):
    # 1) intersect genes (order is preserved by AnnData slicing)
    common_genes = gt_adata.var_names.intersection(pred_adata.var_names)

    ## Setting it to only those genes that the model generates to prevent lookup inflation
    common_genes = ['Prkcq', 'Syt6', 'Ptprm', 'Hspg2', 'Cxcl14', 'Dock5', 'Stxbp6', 'Nfib', 'Gfap', 'Gja1', 'Tcf7l2', 'Rorb', 'Aqp4', 'Slc7a10', 'Grm3', 'Slc1a3', 'Serpine2', 'Lgr6', 'Slc32a1', 'Adamts19', 'Cdh20', 'Sox2', 'Lpar1', 'Pcp4l1', 'Spock3', 'Lypd1', 'Zeb2', 'Unc13c', 'Rgs6', 'Sox6', 'Tafa2', 'Lrp4', 'St6galnac5', 'C030029H02Rik', 'Ust', '2900052N01Rik', 'Sp8', 'Igf2', 'Fli1', 'Opalin', 'Sox10', 'Acta2', 'Chrm2', 'Gad2', 'Cgnl1', 'Vcan', 'Cldn5', 'Mog', 'Maf', 'Bmp4', 'Ctss', 'Dach1', 'Grm8', 'Zfp536', 'Zic1', 'Bcl11b', 'Prkd1', 'C1ql1', 'Hs3st4', 'Pdgfd', 'Nxph1', 'Ebf1', 'Klk6', 'Man1a', 'Sema3c', 'Nr2f2', 'Tgfbr2', 'Pde3a', 'Zfpm2', 'C1ql3', 'Marcksl1', 'Gli2', 'Sema5a', 'Wls', 'Hmcn1', 'Abcc9', 'Kcnip1', 'Mecom', 'Tshz2', 'Nfix', 'Gli3', 'Meis1', 'Kcnmb2', 'Egfem1', 'Adamtsl1', 'Tbx3', 'Gfra1', 'Fign', 'Glis3', 'Kcnj8', 'Adgrf5', 'Vip', 'Chn2', 'Tafa1', 'Ntng1', 'Grik1', 'St18', 'Rmst', 'Dscaml1', 'Synpr', 'Adra1a', 'Prom1', 'Cpa6', 'Sox17', 'Gm37587', 'Prex2', 'Kcnb2', 'Gm28376', 'Il17a', 'Mcm3', 'Paqr8', 'Kcnq5', 'Gm597', 'Ptpn18', 'Gm28417', 'Prss39', 'Neurl3', 'Fam178b', 'Map4k4', 'Gm28782', 'Col3a1', 'Tmeff2', 'Gm28322', 'Rftn2', 'Gm10561', 'Plcl1', 'Aox1', 'Casp8', 'Pard3b', 'Crygc', 'Erbb4', 'Ikzf2', 'Fn1', 'Gm29183', 'Tmbim1', 'Slc11a1', 'Ctdsp1', 'Cyp27a1', 'Des', 'Pax3', 'Fam124b', 'Dock10', 'Gm6217', 'Krtap28-13', 'Pid1', 'Dner', 'Sp110', 'Gm16094', 'Gm10552', 'Sp100', 'A630001G21Rik', 'Ptma', 'Efhd1', 'Inpp5d', 'Ugt1a7c', 'Ugt1a6a', 'Ugt1a5', 'Glrp1', 'Gm29336', 'Agap1', 'Mlph', 'Neu4', 'Gm7967', 'Phlpp1', 'Cdh19', 'Cntnap5a', 'Gm26831', 'Gm29455', 'Dbi', 'Celrr', 'Dpp10', 'Nckap5', 'Tmem163', 'Pm20d1', 'Slc45a3', 'F730311O21Rik', 'Cntn2', 'Prelp', 'Btg2', 'Gpr37l1', 'Csrp1', 'Phlda3', 'Tnni1', 'Mroh3', 'Ptprc', 'Nek7', 'Cfh', 'Rgs1', 'Brinp3', 'Ncf2', 'E330020D12Rik', 'Glul', 'Gm29290', 'Gm29291', 'Gm28286', 'A830008E24Rik', 'Gm37294', '4930523C07Rik', 'Gm31925', 'Myoc', 'Fmo2', 'Prrx1', 'Sell', 'Xcl1', 'Mpzl1', 'Rcsd1', 'Gm37073', 'Gpa33', 'Rgs5', 'Ccdc190', 'Fcgr2b', 'Fcgr3', 'Fcer1g', 'Adamts4', 'Arhgap30', 'F11r', 'Cd48', 'Cd84', 'Atp1a2', 'Kcnj10', 'Slamf9', 'Tagln2', 'Ifi214', 'Ifi213', 'Ifi209', 'Ifi208', 'Ifi207', 'Ifi204', 'Ifi211', 'Ifi203', 'Or6k2', 'Tmem63a', '2210411M09Rik', '1700047M11Rik', 'Trp53bp2', 'Gm37885', 'Tlr5', 'Gm37800', 'Hlx', 'Gm37662', 'Prox1', 'Prox1os', 'Garin4', 'Atf3', 'Traf3ip3', 'Itih5', 'Gm38171', '8030442B05Rik', 'Gm13293', 'Gm13291', 'Il2ra', 'Vim', 'Slc39a12', 'Gm13266', 'Plxdc2', 'Gm13322', 'Pip4k2a', '1810010K12Rik', 'Etl4', 'Apbb1ip', 'Il1rn', 'Slc34a3', 'Sapcd2', 'Entpd2', 'Ptgds', 'Lcn9', 'Nacc2', 'Gpsm1', 'Notch1', 'Lcn4', 'Gm10134', 'Ralgds', 'Pkn3', 'Fnbp1', 'Lamc3', 'Aif1l', 'Lcn2', 'Eng', 'Gm13536', 'Gm13403', 'Pbx3', 'Gsn', 'Ggta1', 'Olfml2a', 'Lrp1b', 'Gm13479', 'Gm13470', 'Tnfaip6', 'Gm33594', 'Galnt5', 'Ermn', 'Gm13546', 'Tanc1', 'Pla2r1', 'Gm13583', 'Grb14', 'Cobll1', 'Slc38a11', 'Nostrin', 'Gad1', 'Dcaf17', 'Itga6', 'Wipf1', 'Gm13708', 'Hoxd4', 'Haglr', 'Gm14424', 'Hoxd1', 'Nfe2l2', 'E030042O20Rik', 'Gm13657', 'Ccdc141', 'Itgav', 'Calcrl', 'Or5m11b', 'Or5m5', 'Pramel6', 'Or5w8', 'Or4b13', 'Spi1', 'Cd82', 'Gm10804', 'Gm13889', 'Prr5l', 'Gm13905', 'Ldlrad3', 'Slc1a2', 'Gm13869', 'Abtb2', 'Ano3', 'Meis2', 'Bmf', 'Gm14207', 'Stard9', 'Frmd5', 'B2m', 'Gatm', 'Sqor', 'Sema6d', 'Itpripl1', 'Mal', 'Mall', 'Morrbid', 'Mertk', 'Gm14011', 'Il1a', 'Rassf2', 'Rrbp1', 'Rin2', 'Nkx2-2', 'Nkx2-2os', '6430503K07Rik', 'Gm14110', 'Thbd', 'Cd93', 'Cst3', 'Cst7', 'Acss1', 'Slc52a3', 'Rem1', 'Id1', 'Cox4i2', 'Foxs1', 'Hck', '2310005A03Rik', 'Procr', 'Myl9', 'Mtcl2', 'Nnat', 'Tgm2', 'Ppp1r16b', 'Sgk2', 'Sdc4', 'Dbndd2', 'Wfdc6a', 'Pltp', 'Slc2a10', 'Prex1', 'Snai1', 'Gm11476', 'Gm14321', '1200007C13Rik', 'Bcas1', 'Bcas1os2', 'Edn3', 'Gm38335', 'Car13', 'Car2', 'Cp', 'Tnfsf10', 'Pld1', 'Gm15462', 'Cldn11', 'Pex5l', 'Sox2ot', 'Gm20515', 'Gm42205', 'Gm12532', 'Gm12534', '5430434I15Rik', 'Gm36823', 'Gm43538', 'Elf2', 'Maml3', 'Tm4sf1', 'Ankub1', 'Rnf13', 'P2ry13', 'P2ry12', 'Trim59', 'Gm20754', 'Gm3513', 'Tlr2', 'Sh3d19', 'Gm37876', 'Fcrl2', 'Fcrl1', 'Pear1', 'Nes', 'Bcan', 'Hapln2', 'Paqr6', 'She', 'Snapin', 'S100a1', 'S100a13', 'S100a14', 'S100a16', 'S100a9', 'Sprr3', 'S100a11', 'Selenbp1', 'Tnfaip8l2', 'Cers2', 'Adamtsl4', 'BC028528', 'Car14', 'Otud7b', 'Gm15444', 'Fcgr1', 'Gm15441', 'Txnip', 'Fmo5', 'Notch2', 'Hmgcs2', 'Gm43189', 'Phgdh', '4930406D18Rik', 'Gm43121', 'Gm42868', 'Igsf3', 'Casq2', 'Gm43241', 'A230001M10Rik', 'Gm43242', 'Gm42682', 'Tspan2', 'Olfml3', 'Gm15471', 'Gm40117', 'Rap1a', 'Gm5547', 'Adora3', 'Cd53', 'Ubl4b', 'Alx3', 'Gstm1', 'Stxbp3', 'S1pr1', 'Gm9889', 'Vcam1', 'Dpyd', 'Cnn3', 'A730020M07Rik', 'F3', 'Ugt8a', 'D030025E07Rik', 'Enpep', 'Etnppl', 'Bdh2', 'Emcn', 'Dapp1', 'Gm19708', '4930425O10Rik', 'Unc5c', 'Pdlim5', 'Gbp5', 'Gbp3', 'Gbp2', 'Gm16233', 'Clca4a', 'Gm35066', 'Gng5', '4930555A03Rik', 'Adgrl4', 'Ifi44', 'St6galnac3', 'Rpe65', '4930430E12Rik', 'Chd7', 'Gm11827', 'Nkain3', 'Bach2os', 'Mob3b', 'Lingo2', 'Myorg', 'Hrct1', 'Or13c7b', 'Tmod1', 'Tgfbr1', 'Grin3a', 'Abca1', 'Slc44a1', 'Klf4', 'Gm12530', 'E130308A19Rik', 'Akna', 'Tlr4', 'Slc24a2', 'Gm13283', 'Gm13272', 'Ifnz', 'Gm13285', 'Gm13288', 'Cdkn2b', 'Jun', 'Cyp2j12', 'Cyp2j6', 'Cyp2j9', 'Nfia', 'Gm12688', 'Foxd3', 'Ube2uos', 'Pde4b', 'Plpp3', 'Calr4', '9630013D21Rik', 'Cdkn2c', 'Prdx1', 'Tie1', 'Tmem125', 'Gm12863', 'Lao1', 'Gm12862', 'Ermap', 'Ybx1', 'Mfsd2a', 'Heyl', 'Fhl3', 'Zc3h12a', 'Csf3r', 'Eva1b', 'Gja4', 'Tlr12', 'Fam167b', 'Tinagl1', 'Nkain1', 'Laptm5', 'Ptafr', 'Themis2', 'Wasf2', 'Ldlrap1', 'Clic4', 'Id3', 'C1qb', 'C1qc', 'C1qa', 'Alpl', 'Klhdc7a', 'Arhgef19', 'Gm13056', 'Epha2', 'Srarp', 'Tmem82', 'Tmem51', 'Tmem51os1', 'Pdpn', 'Pramel14', 'Tnfrsf1b', 'Plod1', 'Gm13073', 'Slc2a5', 'Vamp3', 'Gm13174', 'B230104I21Rik', 'Gm13133', 'Gm13134', 'Prdm16os', 'Gm27202', 'Gm13110', 'Actrt2', 'Tnfrsf14', 'Tmem88b', 'Isg15', 'Gm42435', 'Zfp804b', 'Abcb1a', '1700108N06Rik', 'Sema3d', 'Lhfpl3', 'Hycc1', 'Asb10', 'Gm21655', 'Gm21663', 'Gm1979', 'Insig1', 'Il6', '3110082J24Rik', 'Fgfr3', 'Rgs12', 'Htra3', 'Sh3tc1', 'Msx1', 'Fgfbp1', 'Qdpr', 'Gm42413', 'Sod3', 'C030018K13Rik', 'Klf3', 'Tlr1', 'Tlr6', 'Klhl5', 'Rhoh', 'Rbm47', 'Limch1', 'Fryl', 'Spata18', 'Pdgfra', 'Gm42802', 'Kdr', 'Tecrl', 'Gm43567', 'Slc4a4', 'Npffr2', 'Pf4', 'Cxcl1', 'Cxcl2', 'Cxcl10', 'Gm20500', 'Anxa3', 'Gm33100', 'Gm33370', 'Gm8013', 'Bmp2k', 'Nkx6-1', '5430427N15Rik', 'Sparcl1', 'Gbp8', 'Gbp9', 'Gbp4', 'Gm26519', 'Gm32051', 'Ttc28', 'Cryba4', 'Crybb1', 'Ccdc121rt3', 'Cmklr1', 'Gm15736', 'Tmem119', 'Selplg', 'Gltp', 'Oasl2', 'Oasl1', 'Rplp0', 'Tbx3os1', 'Gm16063', 'Gm16064', 'Oas1g', 'Oas1a', '1700008B11Rik', 'Hvcn1', 'Hcar2', 'Hcar1', 'Ubc', 'Gm42495', 'Gm42500', 'Ncf1', 'Mlxipl', 'Hip1', 'Ccl24', 'Hspb1', 'Cux1', 'Trim56', 'Ephb4', 'Pcolce', 'Tsc22d4', 'Pilra', 'Pilrb1', 'Pilrb2', 'Cyp3a13', 'Gjc3', 'Pvrig', 'Gper1', 'Grifin', 'Gm43703', 'Lfng', 'Gna12', 'Mmd2', 'Actb', 'Arpc1a', 'Arpc1b', 'Gm3404', 'Gsx1', 'D5Ertd605e', 'Flt1', '2210417A02Rik', 'Stard13', 'Gm8579', 'Gng11', 'Col1a2', 'Pon3', 'Pon2', 'Pdk4', 'Slc25a13', 'Dlx6os1', 'Thsd7a', 'Ptprz1', 'Aass', 'Slc13a1', 'Tmem229a', 'Gpr37', 'Gm26627', 'Irf5', 'Plxna4os1', 'Cald1', 'Gm13861', 'Tmem140', 'Slc13a4', 'Fam180a', 'Ptn', 'Zc3hav1', 'Hipk2', 'Gm42962', 'Dennd2a', 'Clec5a', 'Trbv19', 'Trbj1-4', 'Gm44731', 'Sspo', 'Rarres2', 'Gimap8', 'Gimap9', 'Gimap4', 'Gimap6', 'Gimap7', 'Gimap1', 'Gimap5', '4833403J16Rik', 'Gimap3', 'Aoc1l2', 'Gm44109', 'Hoxa1', 'Hotairm1', 'Hoxaas2', 'Gm15050', 'Gm29430', 'Creb5', 'Tril', 'Gm16499', 'Gm16499-1', 'Nod1', 'Inmt', 'Pde1c', 'Grid2', 'Gng12', 'Igkv4-61', 'Krcc1', 'Vamp5', 'Vamp8', 'Capg', 'Tcf7l1', 'Reg3b', 'Reg3a', 'Lrrtm4', 'Gm20383', 'Gm44202', 'Eva1a', 'Gm15864', 'Hk2', 'Gm38843', 'Gm32591', 'Dok1', 'Loxl3', 'Lbx2', 'Nat8f6', 'Nat8f5', 'Nat8', 'Tgfa', 'Gm43936', 'Gp9', 'Gata2', 'Frmd4b', 'Mitf', '1700049E22Rik', 'Lrrn1', 'Gm43964', 'Slc6a11', 'Slc6a1', 'Vgll4', 'Alox5', 'Depp1', 'Gm9946', 'Adipor2', 'Wnk1', 'Ninj2', 'Slc25a18', 'Usp18', 'Slc6a13', 'Slc6a12', 'Apobec1', 'C3ar1', 'Clec4a1', 'Clec4a3', 'Clec4a2', 'Clec4n', 'Clec4d', 'Ltbr', 'Tnfrsf1a', 'Cd9', 'Gm28809', 'Gm26728', 'Vwf', 'Dyrk4', 'Tex52', 'Gm44067', 'BC035044', 'Clec2i', 'Clec2d', 'Cd69', 'Clec1b', 'Klrd1', 'Klrc2', 'Klrc1', 'Klri1', 'Ptpro', 'Mgst1', 'Slco1c1', 'Slco1b2', 'Slco1a4', '5330439B14Rik', 'Rassf8', 'Bhlhe41', 'Itpr2', 'Rps9', 'Pira2', 'Lilra5', 'Ncr1', 'C5ar2', 'C5ar1', 'Slc1a5', 'Six5', 'Apoc1', 'Apoe', 'Gm16174', 'Vmn1r176', 'Ceacam1', 'Tgfb1', 'Axl', 'Plekhg2', 'Zfp36', 'Gm44710', 'Gmfg', 'Sirt2', 'Ppp1r14a', 'Gm29326', 'A330087D11Rik', 'Tyrobp', 'Nphs1os', 'Mag', 'Gm44662', 'Lsr', 'Garre1', 'Plekhf1', 'Lim2', 'Nkg7', 'Siglecl1', 'Cd33', 'Siglece', 'Josd2', '2310016G11Rik', 'Bcl2l12', 'Rcn3', 'Fcgrt', 'Cd37', 'Kcna7', 'Ftl1', 'Emp3', 'Gm45442', 'Saa1', 'Saa2', 'Luzp2', 'Siglech', 'Peg12', 'Klf13', 'Gm32633', 'Pcsk6', 'Lrrk1', 'Gm44752', 'Gm44725', 'Gm20083', 'Gm30075', 'Rgma', 'AU020206', 'Mfge8', 'Rlbp1', 'Anpep', 'Pde8a', 'Folh1', 'Ctsc', 'Serpinh1', 'Slco2b1', 'Gm15635', 'Gm34280', 'Plekhb1', 'P2ry6', 'Gm45620', 'Rhog', 'Or51e1', 'Or52e2', 'Hbb-bt', 'Olfm5', 'Trim34a', 'Gm47248', 'Trim5', 'Trim12a', 'Gm15133', 'Trim12c', 'Trim30b', 'Trim30d', 'Cavin3', 'Gvin2', 'Gvin1', 'Olfml1', 'Gm44773', 'Ppfibp2', 'Nrip3', 'Insc', 'Itpripl2', 'Ccp110', 'Gprc5b', 'Acsm5', 'Gm44866', 'Igsf6', 'Chp2', 'Il21r', 'Sult1a1', 'Kctd13', 'Itgam', 'Itgad', 'Rgs10', 'Gm33027', 'Fgfr2', 'Htra1', 'D7Ertd443e', 'Dock1', 'Gm36356', 'C230079O03Rik', 'Gm36849', 'Gm6249', 'Nkx6-2', 'Ifitm1', 'Ifitm3', 'Sigirr', 'Irf7', 'B230206H07Rik', 'Cracr2b', 'Tnni2', 'Igf2os', 'Tspan32', 'Cd81', 'Ano1', 'Gm44929', 'Mrgprf', 'Cd209f', 'Myo16', 'Rab20', 'E230013L22Rik', 'Lamp1', 'Arhgef10', 'Gm16350', 'Defb12', 'Defb10', 'Zmat4', 'Tcim', 'Adam3', 'Plekha2', 'Gm32098', 'Nrg1', 'Rbpms', 'Gm26632', 'Gm26978', 'Ppp1r3b', 'Gm38414', 'Dlc1', 'Slc7a2', 'Mtus1', 'Gm16192', 'Tlr3', 'Enpp6', 'Tenm3', 'C130073E24Rik', 'Hdnr', 'Scrg1', 'Galntl6', 'Marchf1', 'Gm32568', 'Nat2', 'Uba52', 'Lrrc25', 'Jund', 'Bst2', 'Tmem221', 'Slc27a1', 'Insl3', 'Gm35572', 'Klf2', 'Nwd1', 'Nr3c2', 'Ednra', 'Gm30329', 'Hhip', 'Gab1', 'Gm31105', 'Gm27048', 'Inpp4b', 'Il15', 'Adgre5', 'Lyl1', 'Gm42031', 'Gm27168', 'Nkd1', 'Sall1', 'Gm36325', 'Mmp2', 'Mt2', 'Mt1', 'Gm15889', '9330175E14Rik', 'Pllp', 'Cdh5', 'Cmtm3', 'Ces2e', 'Cbfb', 'Lcat', '6030452D12Rik', 'Zfhx3', '8030455M16Rik', 'Mlkl', 'Fa2h', 'Gm16118', 'Osgin1', 'Gm32352', 'Wfdc1', 'Fendrr', 'Foxf1', 'Foxc2', '5033426O07Rik', 'Gm26747', 'Gm26784', 'Car5a', 'Cyba', 'Abcb10', 'Fam89a', 'Pard3', 'Casp4', 'Birc3', 'Cntn5', 'Maml2', 'Naalad2', 'Icam1', 'S1pr5', 'Anln', 'Gm48646', 'Cypt4', 'Jam3', 'Gm27201', 'Ets1', 'Gm3331', 'Gm47784', '7630403G23Rik', 'Kirrel3', 'St3gal4', 'Gm48081', 'Hepacam', 'Robo4', 'Nrgn', 'Or8c18', 'Ubash3b', 'Oaf', 'D630033O11Rik', 'Gm36855', 'C1qtnf5', 'Gm10687', 'Mcam', 'Phldb1', 'Cd3g', 'Cd3e', 'Il10ra', 'Tagln', 'Sik3', 'Bco2', 'Il18', 'Hspb2', 'Cryab', 'Rdx', 'Acsbg1', 'Cspg4', 'Snx33', 'Uaca', 'Gm39348', 'Paqr5', 'Scarletltr', 'Rasl12', 'Plekho2', 'Rbpms2', 'Snx22', 'Rps27l', 'Rora', 'Aqp9', 'Tcf12', 'Elovl5', 'Cilk1', 'Myo6', 'Tbx18', 'Bcl2a1d', 'Bcl2a1a', 'Bcl2a1b', 'Ctsh', 'Gm31409', 'Plscr1', 'Plscr2', 'Slc9a9', 'Pls1', 'Atp1b3', 'Rnf7', 'Foxl2', 'Trf', 'Bfsp2', 'Cpne4', 'Gnai2', 'Slc38a3', 'Lamb2', 'Gm7628', 'Gm42756', 'Fbxw15', 'Cripto', '1700061E17Rik', 'Ccrl2', 'Bcl2a1c', 'C130032M10Rik', 'Rbms3', 'Itga9', 'Cx3cr1', 'Xirp1', 'Gm33858', 'Mobp', 'Cck', 'Slc6a20a', 'Cxcr6', 'Ccr1', 'Ccr2', 'Ccr5', 'Plekhg1', 'Myct1', 'Gm48614', 'Sash1', 'Utrn', 'Phactr2', 'Gm20139', 'Gm33104', 'Map7', 'Pde7b', '1700020N01Rik', 'Gm5420', 'Tcf21', 'Epb41l2', 'Lama2', '4930579H20Rik', 'Ptprk', 'Frk', 'Gm26535', 'Cdk19', 'Gm48061', 'Gm48065', 'Gm35154', 'Gm48066', 'Gm47389', 'Gm35552', 'Gm46189', 'Lilrb4b', 'Gm40645', 'Lilrb4a', 'Gm46224', 'Gm40650', 'Fabp7', 'Ddit4', 'Chst3', 'Cdh23', 'Vsir', 'Unc5b', 'Nodal', 'Tspan15', 'Srgn', 'Gm16143', 'Ctnna3', 'Reep3', 'Ado', 'Gm33979', 'Gm34776', 'A330049N07Rik', 'Pcdh15', 'Adora2a', 'Ggt1', 'Ggt5', 'Gstt3', 'S100b', 'Pcbp3', 'Itgb2', 'Hcn2', 'Cnn2', 'Arhgap45', 'S1pr4', 'Gna15', 'Appl2', 'Rfx4', 'Fbxo7', 'Timp3', 'Ano4', 'Gm32688', 'Tmcc3', '1110019B22Rik', 'Btg1', 'Galnt4', 'Gm35035', 'Mgat4c', 'Rassf9', 'Alx1', 'Tmtc2', 'Phlda1', 'Kcnc2', 'Lgr5', 'A930009A15Rik', 'Lyz2', 'Cpm', 'Gm40773', 'Grip1', 'Srgap1', 'Arhgap9', 'Gli1', 'Ndufa4l2', 'Erbb3', 'Cdk2', 'Cd63', 'Pla2g3', 'Inpp5j', 'Selenom', 'Tcn2', 'Gal3st1', 'Castor1', 'Osm', 'Lif', 'Aebp1', 'Tns3', 'Gm11998', 'Gm11999', 'Ikzf1', 'Hba-a1', 'Kcnmb1', 'Dock2', 'Gm12145', 'Gm12148', 'Gm12158', 'Nipal4', 'Havcr2', 'Sgcd', 'Gm12185', 'Psme2b', '9930111J21Rik1', 'Tgtp1', '9930111J21Rik2', 'Tgtp2', 'Ifi47', 'Scgb3a1', 'Ltc4s', 'Pdlim4', 'Slc36a2', 'Sparc', 'Irgm2', 'Gjc2', 'Nlrp3', 'Tnfrsf13b', 'Grap', 'Adora2b', 'Cdrt4os1', 'Pmp22', 'Kdm6b', 'Dnah2os', 'Efnb3', 'Atp1b2', 'Shbg', 'Tnfsf13', 'Kctd11', 'Slc2a4', 'Asgr2', 'Cxcl16', 'Gm40193', 'Nlrp1a', 'Nlrp1b', 'Gm16013', 'Gm12324', 'Gm12326', 'Wscd1', 'Ggt6', 'Atp2a3', 'Aspa', 'Hic1', 'Tlcd1', 'Aldoc', 'Vtn', 'Evi2', 'Evi2a', 'Adap2', 'Adap2os', 'Myo1d', 'Tmem98', 'Ccl2', 'Ccl7', 'Ccl11', 'Ccl12', 'Ccl8', 'Rffl', 'Unc45b', 'Slfn5', 'Slfn8', 'Slfn1', 'Slfn3', 'Mmp28', 'Ccl5', 'Ccl9', 'Ccl6', 'Ccl3', 'Ccl4', 'Tbx2', 'Vmp1', 'Septin4', 'Mir142hg', 'Cuedc1', 'Msi2', 'Trim25', 'Tmem100', 'Kif2b', 'Abcc3', 'Acsf2', 'Col1a1', 'H1f9', 'Abi3', 'Gngt2', 'Gm11535', 'Hoxb4', 'Arhgap23', 'Jup', 'Cnp', 'Cavin1', 'Gm11627', 'Fzd2', 'Higd1b', 'Arhgap27os3', 'Cd79b', 'Icam2', 'Pitpnc1', 'Cacng4', 'Prkca', 'E030025P04Rik', 'Arsg', '1700023C21Rik', 'Abca9', 'Gm11674', 'Gm11681', '2610035D17Rik', 'Ttyh2', 'Gprc5c', 'Cd300a', 'Cd300c', 'Cd300ld', 'Cd300c2', 'Cd300ld3', 'Caskin2', 'Smim5', 'Itgb4', 'H3f3b', 'Rhbdf2', '6030468B19Rik', 'Syngr2', 'Notum', 'Cbr2', 'Gm48678', 'Gm48071', 'Rhob', 'Ntsr2', 'Id2', 'Gm47705', 'Gm47713', 'Rsad2', 'Gm28806', 'Pik3cg', 'Agmo', 'Dgkb', 'Etv1', 'Gm48508', 'Gm35135', 'Gm35188', 'Npas3', 'Gm7550', 'Nfkbia', 'Nkx2-9', 'Six1', 'Gm39473', 'Rhoj', 'Plekhg3', 'Gm35189', 'Plekhh1', 'Gm47752', 'Zfp36l1', 'Npc2', 'Fos', 'Gm32296', 'Batf', 'Flvcr2', '1700040E09Rik', 'Ston2', 'Foxn3', 'Kcnk13', 'Lgmn', 'Gm30198', 'Prima1', 'Gm15523', 'Clmn', 'Gm36757', 'Pld4', 'Pacs2', 'Crip1', 'Ighj4', 'Itgb8', 'Akr1c14', 'Gm47904', 'Akr1c13', 'Akr1c12', 'Adarb2', 'Nid1', 'Gng4', 'Trgv4', 'Trgv6', 'Trgc1', 'Trgc2', 'Trgj2', 'Elmo1', 'H2bc15', 'H2bc24', 'H4c8', 'H2bc8', 'H2bc4', 'Hfe', 'H1f2', 'Sox4', 'Gm11368', 'Foxq1', 'Gm11378', 'Foxf2', 'Foxc1', 'Gm11381', 'Gm48073', 'Serpinb1a', 'Ppp1r3g', 'Ly86', '2210022D18Rik', 'Gm47731', 'Gm47754', 'Gm3509', 'Tfap2a', 'Gm48107', 'Gm10790', 'Cd83', 'Kif13a', 'A330048O09Rik', 'Susd3', 'Aspn', 'Ogn', 'Msx2', 'Gm16578', 'Tifab', 'Neurog1', 'Tgfbi', 'Gkap1', 'Gm48384', '4930486L24Rik', 'Ctla2b', 'Ctla2a', 'Slc35d2', 'Arrdc3', 'Gm49375', 'Mef2c', 'Edil3', 'Serinc5', 'Gm48287', 'Hexb', 'Foxd1', 'Gm807', 'Zfp366', 'Naip2', 'Naip5', 'Naip6', 'Pik3r1', 'Gm47007', 'Gm29927', 'Cd180', 'Mast4', 'Erbin', 'Gm30411', 'Elovl7', 'Map3k1', 'Gm15323', '2810403G07Rik', 'Gm48876', 'Gm48879', 'Gpx8', 'Gm41071', 'Itga1', 'BC147527', 'Gm3629', 'Gm8362', 'Fam107a', 'Gm48370', 'Gm48371', 'Sntn', 'Nid2', 'Kcnk5', 'Usp54', 'Lrmda', 'Gm47601', 'E330034G19Rik', 'Zcchc24', 'Chdh', 'Itih3', 'Stab1', 'Tmem273', 'Vstm4', 'Gm49030', 'Zfp488', 'Fermt2', 'Gm49303', 'Gm49305', 'Gm49306', 'Peli2', 'Rnase12', 'Rnase4', 'Ang', '1810028F09Rik', 'Rnase1', 'Ndrg2', 'Gm49076', 'Or10g3', 'Or10g1b', 'Trav11', 'Trav7-5', 'Trav3-3', 'Trav15-3', 'Trav3-4', 'Trav12-4', 'Traj24', 'Traj4', 'Slc7a7', 'Mmp14', 'Ajuba', 'Cmtm5', 'Gm49130', 'Adcy4', 'Ripk3', 'Nfatc4', 'Gzmb', 'Gjb2', 'Gm4491', 'Gjb6', 'Zdhhc20', 'Rcbtb1', 'Phf11b', 'Arl11', 'Dleu2', 'Kcnrg', 'Defb47', 'Sox7', 'Kif13b', 'Pdlim2', 'Sorbs3', 'Cysltr2', 'Lpar6', 'Tpt1', 'Kctd4', '2900040C04Rik', 'Gm4632', 'Rgcc', 'Elf1', 'Gm49042', 'Pcdh17', '9630013A20Rik', 'Gm34589', 'Slain1', 'Ednrb', 'Gm6145', 'Gm49010', '1700100I10Rik', 'Gm31072', 'Gpc5', 'Gpc6', 'Sox21', 'Sox21os1', 'Gm9376', 'Gm32093', 'Hs6st3', 'Gpr183', 'Gm5089-1', 'Selenop', 'Fyb1', 'Osmr', 'Il7r', 'Cdh10', 'Cdh18', '9630009A06Rik', 'Cpq', 'Gm16136', 'Gm26704', 'Fzd6', 'Gm49786', 'Trps1', 'Enpp2', 'Mtss1', 'Tmem71', 'Ndrg1', 'Gpr20', 'Lypd2', 'Ly6i', 'Ly6a', 'Ly6c1', 'Gsdmd', 'Eef1d', 'Apol10b', 'Ncf4', 'Csf2rb2', 'Csf2rb', 'Tst', 'Rac2', 'Gm36738', 'Cyth4', 'Mfng', 'Cdc42ep1', 'Lgals2', 'Gm10863', 'Apobec3', 'Gm17025', 'Desi1', 'Shisa8', 'Cyp2d22', 'Bik', 'Tspo', '5031439G07Rik', 'Ppara', 'Gm15722', 'Pim3', 'Ttll8', 'Mlc1', 'Gm41386', 'Slc38a2', 'Or10ad1', 'Tuba1c', 'Smagp', 'Bin2', 'Galnt6', 'Galnt6os', 'I730030J21Rik', 'Acvrl1', 'Tns2', 'Sp7', 'Gpr84', 'Nckap1l', 'Mucl2', 'Sec14l5', 'AU021092', 'Carhsp1', 'Litaf', 'Tnfrsf17', 'Gm4279', 'Nde1', 'Myh11', 'Tbx1', 'Iglc1', 'Iglc3', 'Iglc2', 'Klhl6', 'St6gal1', 'Sst', 'Gm46545', 'Lpp', 'Gm4524', 'Atp13a5', 'Atp13a4', 'Apod', 'Gm34256', 'Heg1', 'Itgb5', 'Parp14', 'Cd86', 'Hcls1', 'Pla1a', 'Arhgap31', 'Upk1b', 'Igsf11', 'Gm36742', 'Zbtb20', 'Cd200l1', 'Phldb2', 'Gm15640', 'Cep97', 'Tmem45a', 'Filip1l', 'St3gal6', 'Ftdc1', 'Epha6', 'Robo1', 'Samsn1', 'Mir99ahg', 'Ncam2', 'Cyyr1', 'Map3k7cl', 'Gm33255', '4930420G21Rik', 'Cldn17', 'Gm49688', 'Krtap24-1', 'Krtap13-1', 'Gm36001', 'Gm41492', 'Olig2', 'Olig1', 'Runx1', 'Gm49723', 'Cldn14', 'Gm49618', 'Igsf5', 'Pcp4', '3300005D01Rik', 'Tagap', 'Ccr6', '4930506C21Rik', 'Prr18', 'Gm34684', 'Qki', 'A230009B12Rik', 'Gm16168', 'Airn', 'Mmp25', 'Bricd5', 'Tmem204', 'Gm33727', 'Gm50025', 'Sox8', 'Cerox1', 'Metrn', 'Nhlrc4', 'Prr35', 'Gm50275', 'Atp6v0e', 'Gm49793', 'Trp53cor1', 'Fgd2', 'Cbs', '2310015A16Rik', 'Notch3', 'Rasal3', 'Cyp4f15', 'Cyp4f14', 'Myo1f', 'Angptl4', 'H2-K1', 'H2-Oa', 'H2-DMb1', 'Psmb9', 'Psmb8', 'Gm20496', 'H2-Ob', 'H2-Ab1', 'H2-Aa', 'Btnl2', 'Gpsm3', 'C4b', 'Clic1', 'Ly6g6d', 'Ly6g6f', 'Aif1', 'Tnf', 'Gm11131', 'H2-Q4', 'Ddr1', 'Or5v1', 'Ptchd4', 'Ankrd66', 'Pla2g7', '1700071M16Rik', 'Rcan2', 'Mdfi', 'Trem2', 'Daam2', 'Tbc1d5', 'Kat2b', 'Plin4', 'Plin3', 'Tnfsf14', 'Vav1', 'Adgre1', 'Cntnap5c', 'A330072L02Rik', 'Pdzph1', 'Rab31', 'Tgif1', 'Gm26510', '1600022D10Rik', 'Lbh', 'Xdh', 'Rasgrp3', 'Prkd3', 'Cyp1b1', 'Gm49984', '1810073O08Rik', 'Gm35551', 'Zfp36l2', 'Six2', 'Epas1', 'Svil', 'Zeb1', 'Npc1', 'Gm2629', 'Gm50035', 'BC051408', 'Gm33948', 'Lims2', 'Gpr17', 'Bin1', 'Ecscr', 'Sting1', 'Cd14', 'Arap3', 'Fgf1', 'Ticam2', 'Sema6a', 'Tnfaip8', 'C030005K06Rik', 'Lox', 'Gm41724', 'Ppic', '9330117O12Rik', 'Zfp608', 'Gramd2b', 'Megf10', 'Gm19500', '2210409D07Rik', 'Slc12a2', 'Chsy3', 'Pdgfrb', 'Csf1r', 'Carmn', 'Sh3tc2', 'Gm41750', 'Adrb2', '2700046A07Rik', 'Rax', 'Tcf4', 'Dcc', 'Smad7', 'Pstpip2', 'Slc14a1', 'Gm41790', 'Nfatc1', 'Gm50211', 'Sall3', 'Gm27239', 'Mbp', 'Tshz1', 'Cpt1a', 'Tesmin', 'Unc93b1', 'Cd248', 'Neat1', 'Frmd8', 'Slc25a45', 'Gm14964', 'Plaat3', 'Slc22a8', 'Slc22a6', 'Asrgl1', 'Gm50321', 'Myrf', 'Gm28347', 'AW112010', 'Ms4a6c', 'Ms4a6b', 'Ms4a6d', 'Mpeg1', 'Psat1', 'Gcnt1', 'Ostf1', '1500015L24Rik', '4930554I06Rik', 'Abhd17b', 'Gm50130', 'Trpm3', 'Tmem252', 'Dock8', 'Kank1', 'Gm9895', 'Il33', 'Prkg1', 'Fas', 'Hhex', 'Plce1', 'Gm28991', 'Entpd1', 'Pik3ap1', 'Sfrp5', 'Pkd2l1', 'Gm50323', 'Scd2', 'Scd1', 'Lzts2', 'Sorcs1', 'Afap1l2', 'Pnlip', 'Was', 'Slc38a5', 'H2al1k', 'Otc', 'Mid1ip1', 'Gpr34', 'Dipk2b', 'Gm2309', 'Apln', 'Sash3', 'Elf4', 'Frmd7', 'Mir503hg', 'Mir503hg-1', 'Zfp36l3', 'Xlr', '3830403N18Rik', 'F9', '4931400O07Rik', 'Bgn', 'Plxnb3', 'Renbp', 'Ctag2l1', 'Tasl', 'Msn', 'F630028O10Rik', 'Heph', 'Gjb1', 'Cxcr3', 'Phka1', 'Xist', 'Magt1', 'Tlr13', 'Cysltr1', 'Lpar4', 'P2ry10', 'P2ry10b', 'Itm2a', 'Pcdh11x', 'Plp1', 'Il1rapl2', 'Tsc22d3', 'Col4a6', 'Col4a5', 'Gucy2f', 'Kcne5', 'Htr2c', 'Il13ra2', 'Cldn34b2', 'Sat1', 'Nhs', 'Ap1s2', 'Cltrn', 'Ace2', 'Gpm6b', 'Gm15232', 'Tlr7', 'Gm29044', 'Gm37572', 'Rbm31y', 'Gm29343', 'Gm29003', 'Gm28898', 'Gm29392', 'mt-Nd3', 'mt-Nd4l']
    common_genes_atleast1 = ['Prkcq', 'Syt6', 'Ptprm', 'Hspg2', 'Cxcl14', 'Dock5', 'Stxbp6', 'Nfib', 'Gfap', 'Gja1', 'Tcf7l2', 'Rorb', 'Aqp4', 'Slc7a10', 'Grm3', 'Slc1a3', 'Serpine2', 'Lgr6', 'Slc32a1', 'Adamts19', 'Cdh20', 'Sox2', 'Lpar1', 'Pcp4l1', 'Spock3', 'Lypd1', 'Zeb2', 'Unc13c', 'Rgs6', 'Sox6', 'Tafa2', 'Lrp4', 'St6galnac5', 'C030029H02Rik', 'Ust', '2900052N01Rik', 'Sp8', 'Igf2', 'Fli1', 'Opalin', 'Sox10', 'Acta2', 'Chrm2', 'Gad2', 'Cgnl1', 'Vcan', 'Cldn5', 'Mog', 'Maf', 'Bmp4', 'Ctss', 'Dach1', 'Grm8', 'Zfp536', 'Zic1', 'Bcl11b', 'Prkd1', 'C1ql1', 'Hs3st4', 'Pdgfd', 'Nxph1', 'Ebf1', 'Klk6', 'Man1a', 'Sema3c', 'Nr2f2', 'Tgfbr2', 'Pde3a', 'Zfpm2', 'C1ql3', 'Marcksl1', 'Gli2', 'Sema5a', 'Wls', 'Hmcn1', 'Abcc9', 'Kcnip1', 'Mecom', 'Tshz2', 'Nfix', 'Gli3', 'Meis1', 'Kcnmb2', 'Egfem1', 'Adamtsl1', 'Tbx3', 'Gfra1', 'Fign', 'Glis3', 'Kcnj8', 'Adgrf5', 'Vip', 'Chn2', 'Tafa1', 'Ntng1', 'Grik1', 'St18', 'Rmst', 'Dscaml1', 'Synpr', 'Adra1a', 'Prom1', 'Cpa6']
    common_genes = pd.Index(common_genes_atleast1, name="gene")


    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between gt_adata and pred_adata.")
    print(gt_adata.var_names)
    gt_common   = gt_adata[:, common_genes]
    pred_common = pred_adata[:, common_genes]

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
from scipy.stats import pearsonr

def soft_correlation(gt_adata, gt_positions, pred_adata, pred_positions, radius=None, k=0, sample=None):
    """
    gt_expressions, pred_expressions: list or array of gene expression vectors (shape [num_cells, num_genes])
    gt_positions, pred_positions: list or array of positions (shape [num_cells, 2] or [num_cells, 3])
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(gt_adata, pred_adata, 1)
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

        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} samples...")

    gt_sums = np.array(gt_sums).flatten()
    pred_sums = np.array(pred_sums).flatten()

    if len(gt_sums) == 0 or len(pred_sums) == 0:
        return 0.0

    correlation, _ = pearsonr(gt_sums, pred_sums)
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




def soft_f1(gt_adata, gt_positions, pred_adata, pred_positions, radius=None, k=0, sample=None):
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





