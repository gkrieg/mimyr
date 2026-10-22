#!/usr/bin/env python
"""Rescore stage-2 (cell type) predictions at the class level.

The supplement compares MIMYR, the lookup baseline and SpatialZ on cell types in
the only label space all three share: the ~19 ABC-atlas classes. MIMYR predicts
cluster tokens and main.py scores tokens, so this script reads the sections that
supp_stagewise.sh saved (each row's artifact_dir/pred.pkl), maps every token to
its class, and scores exact and soft accuracy.

Stage 2 runs with the true locations (`--location_inference_type skip`), so row i
of a prediction is the same cell as row i of the real section, which is what makes
an exact per-cell accuracy possible.

Usage:
    python reproducibility/analysis/celltype_class_accuracy.py \
        --csv results/reproduce/supp_stage_celltype.csv \
        --out_csv results/reproduce/supp_stage_celltype_class.csv
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from mimyr._resources import META_INFO_FILE
from mimyr.data_loader import SliceDataLoader
from mimyr.metrics import celltype_abundance_metrics, soft_accuracy

METHOD_OF = {"model": "MIMYR", "majority_baseline": "Lookup"}
RADII = [0.03, 0.04, 0.05]
# Abort if more than this share of cells disagrees with its token's majority class.
AMBIGUITY_ABORT_FRAC = 0.005


def token_to_class(sections):
    """token -> class, by majority over the cells of `sections`.

    The map is a tree except for token 51, where the data loader puts clusters
    that have no token ("Reassigned missing tokens to majority class: 51").
    Majority resolves it; the share of cells that disagree is reported.
    """
    df = pd.DataFrame({
        "token": np.concatenate([np.asarray(s.obs["token"]) for s in sections]),
        "cls": np.concatenate([np.asarray(s.obs["class"].astype(str)) for s in sections]),
    })
    counts = df.groupby(["token", "cls"]).size().rename("n").reset_index()
    winner = (counts.sort_values("n", ascending=False)
                    .drop_duplicates("token").set_index("token")["cls"])
    lost = int(counts.merge(winner.rename("win"), left_on="token", right_index=True)
                     .query("cls != win")["n"].sum())
    if lost:
        print(f"  {lost} of {len(df)} cells ({100 * lost / len(df):.4f}%) disagree "
              "with their token's majority class")
    if lost > AMBIGUITY_ABORT_FRAC * len(df):
        raise SystemExit("token -> class is ambiguous for more than 0.5% of cells; "
                         "the class-level comparison is not well defined")
    return winner.to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True, help="stage-2 CSV written by supp_stagewise.sh")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--sample", type=int, default=100,
                    help="percentage of real cells scored by soft accuracy")
    args = ap.parse_args()

    runs = pd.read_csv(args.csv)
    runs = runs[runs["cluster_inference_type"].isin(METHOD_OF)]
    rows = []
    for mode, group in runs.groupby("data_mode", sort=True):
        loader = SliceDataLoader(mode=mode, label="cluster",
                                 cfg={"data_dir": args.data_dir, "meta_info": META_INFO_FILE})
        loader.prepare()
        # The model emits tokens that never occur in the five test sections, so the
        # map is built from the reference sections too; they are adjacent sections
        # under the same annotation.
        tok2cls = token_to_class(list(loader.test_slices) + list(loader.reference_slices or []))

        for run in group.itertuples():
            with open(os.path.join(run.artifact_dir, "pred.pkl"), "rb") as f:
                pred = pickle.load(f)
            gt = loader.test_slices[int(run.slice_index)]
            gt_cls = gt.obs["class"].astype(str).to_numpy()
            pred_tok = np.asarray(pred.obs["token"])
            pred_cls = np.array([tok2cls.get(t, "<unmapped>") for t in pred_tok])
            gt_pos = np.asarray(gt.obsm["aligned_spatial"])[:, :2]
            pred_pos = np.asarray(pred.obsm["spatial"])[:, :2]

            row = {"method": METHOD_OF[run.cluster_inference_type], "data_mode": mode,
                   "slice_index": int(run.slice_index), "n_cells": gt.n_obs,
                   "unmapped_pred_tokens": int((pred_cls == "<unmapped>").sum())}
            if len(pred_cls) == len(gt_cls):
                row["exact_accuracy_class"] = float((pred_cls == gt_cls).mean())
                row["exact_accuracy_token"] = float((pred_tok == np.asarray(gt.obs["token"])).mean())
            for r in RADII:
                row[f"soft_accuracy@{r}"] = soft_accuracy(
                    gt_cls.tolist(), gt_pos, pred_cls.tolist(), pred_pos,
                    radius=r, sample=args.sample)
            vocab = {c: k for k, c in enumerate(sorted(set(gt_cls) | set(pred_cls)))}
            abundance = celltype_abundance_metrics(
                np.array([vocab[c] for c in gt_cls]), np.array([vocab[c] for c in pred_cls]))
            row.update({f"celltype_abundance_{k}": v for k, v in abundance.items()})
            rows.append(row)
            print(f"  {mode} {row['method']:6s} section {row['slice_index']}: "
                  f"exact class {row.get('exact_accuracy_class', float('nan')):.4f}, "
                  f"soft@0.05 {row['soft_accuracy@0.05']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} rows)")
    if len(df):
        print(df.groupby("method")[["exact_accuracy_class", "soft_accuracy@0.05"]].mean().round(4))


if __name__ == "__main__":
    main()
