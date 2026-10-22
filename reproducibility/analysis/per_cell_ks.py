#!/usr/bin/env python
"""Per-cell KS for the supplement's parameter sweeps and cascade ablation.

For every section a sweep generated (each row's artifact_dir/pred.pkl), compare
eight distributions with the real section by the Kolmogorov-Smirnov statistic:
four gene-level ones (mean and variance of log expression, gene detection
frequency, gene-gene correlation) and four cell-level ones (cell log library
size, cell-cell distance, cell detection frequency, cell-cell correlation).
The supplement's "per-cell KS" is the mean over the four cell-level ones
(column mean_ks_cell).

Reads the CSVs written by supp_ref_guidance.sh, supp_decoding.sh,
supp_celltype_temperature.sh and supp_cascade.sh from --results, and writes one
row per generated section to --out_csv.

Usage:
    python reproducibility/analysis/per_cell_ks.py --results results/reproduce
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import euclidean_distances

from mimyr._resources import META_INFO_FILE
from mimyr.data_loader import SliceDataLoader

# (sweep, CSV, whether the gene-level mean/variance panels use log expression)
SWEEPS = [
    ("ref_guidance_rq1", "supp_ref_guidance_rq1.csv", True),
    ("ref_guidance_rq3", "supp_ref_guidance_rq3.csv", False),
    ("decoding", "supp_decoding.csv", False),
    ("argmax", "supp_celltype_argmax.csv", False),
    ("cascade", "supp_cascade.csv", False),
]
PANELS = ["Mean log expression", "Variance of log expression", "Gene detection frequency",
          "Gene-gene correlation", "Cell log library size", "Cell-cell distance",
          "Cell detection frequency", "Cell-cell correlation"]
CELL_PANELS = PANELS[4:]
ID_COLS = ["data_mode", "slice_index", "location_inference_type", "cluster_inference_type",
           "expression_inference_type", "expression_weak_teacher_forcing", "wtf_strength_gene",
           "expression_select_max_index_margin", "top_k", "cluster_temperature",
           "soft_f1_radius@0.05", "soft_spearman_correlation_radius@0.05"]


def distributions(adata, log_mean_var, n_sample=500, seed=0):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    lX = np.log1p(X)
    sub = X[np.random.default_rng(seed).choice(X.shape[0], min(n_sample, X.shape[0]), replace=False)]
    upper = lambda M: M[np.triu_indices(M.shape[0], k=1)]
    mv = lX if log_mean_var else X
    return {
        "Mean log expression": mv.mean(axis=0),
        "Variance of log expression": mv.var(axis=0),
        "Gene detection frequency": (X > 0).mean(axis=0),
        "Gene-gene correlation": upper(np.corrcoef(X.T)),
        "Cell log library size": np.log1p(X.sum(axis=1)),
        "Cell-cell distance": upper(euclidean_distances(sub)),
        "Cell detection frequency": (X > 0).mean(axis=1),
        "Cell-cell correlation": upper(np.corrcoef(sub)),
    }


def ks_panels(pred, gt, log_mean_var):
    genes = gt.var_names.intersection(pred.var_names)
    p = distributions(pred[:, genes].copy(), log_mean_var)
    g = distributions(gt[:, genes].copy(), log_mean_var)
    out = {}
    for panel in PANELS:
        a, b = np.asarray(p[panel]), np.asarray(g[panel])
        out[panel] = ks_2samp(a[np.isfinite(a)], b[np.isfinite(b)]).statistic
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results/reproduce", help="directory with the sweep CSVs")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_csv", default=None, help="default: <results>/supp_per_cell_ks.csv")
    args = ap.parse_args()
    out_csv = args.out_csv or os.path.join(args.results, "supp_per_cell_ks.csv")

    test_sections = {}

    def real_section(mode, i):
        if mode not in test_sections:
            loader = SliceDataLoader(mode=mode, label="cluster",
                                     cfg={"data_dir": args.data_dir, "meta_info": META_INFO_FILE})
            loader.prepare()
            test_sections[mode] = loader.test_slices
        return test_sections[mode][i]

    rows = []
    for sweep, csv, log_mean_var in SWEEPS:
        path = os.path.join(args.results, csv)
        if not os.path.isfile(path):
            print(f"{path} not found; skipping {sweep}")
            continue
        for _, r in pd.read_csv(path).iterrows():
            with open(os.path.join(r.artifact_dir, "pred.pkl"), "rb") as f:
                pred = pickle.load(f)
            ks = ks_panels(pred, real_section(r.data_mode, int(r.slice_index)), log_mean_var)
            row = {"sweep": sweep, "artifact_dir": r.artifact_dir,
                   "mean_ks_all": float(np.mean([ks[p] for p in PANELS])),
                   "mean_ks_cell": float(np.mean([ks[p] for p in CELL_PANELS]))}
            row.update({c: r.get(c) for c in ID_COLS})
            row.update({f"ks::{p}": ks[p] for p in PANELS})
            rows.append(row)
        print(f"done {sweep}", flush=True)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
