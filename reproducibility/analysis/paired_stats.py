#!/usr/bin/env python
"""Paired per-section contrasts quoted in the supplement.

Every sweep holds the five test sections and the random seed fixed and changes
one setting, so any two configurations are matched section by section. For each
contrast this prints mean(arm - reference) +/- its s.e.m., the paired t-test, and
on how many sections the arm is better. n = 5 sections unless stated.

Reads the CSVs written by the supp_*.sh scripts and analysis/per_cell_ks.py from
--results. The stage-wise comparison against SpatialZ also needs SpatialZ's
scores (--spatialz: a CSV with method, data_mode and slice_index columns and the
same metric columns); without it only MIMYR against the lookup baseline is printed.

Usage:
    python reproducibility/analysis/paired_stats.py --results results/reproduce
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

F1 = "soft_f1_radius@0.05"
SP = "soft_spearman_correlation_radius@0.05"


def paired(a, b, label="", invert=False):
    """a, b: dict section -> value. Prints and returns the (arm - reference) contrast."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], float)
    n = len(d)
    m, s = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys])
    npos = int((d > 0).sum())
    # exact two-sided sign test; at n = 5 it cannot go below P = 0.0625
    psign = min(1.0, 2 * stats.binom.sf(max(npos, n - npos) - 1, n, 0.5))
    better = n - npos if invert else npos
    print(f"  {label:<46s} {m:+.4f} +/- {s:.4f}   t={t:+7.2f}  P={p:.2e}  "
          f"{better}/{n} sections improve  P_sign={psign:.4f}")
    return m, s, t, p


def head(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results/reproduce")
    ap.add_argument("--spatialz", default=None,
                    help="SpatialZ stage-wise scores (optional; see the docstring)")
    args = ap.parse_args()
    R = lambda name: os.path.join(args.results, name)

    def tidy(df, armfn, metric):
        """dict arm -> dict section -> metric. Duplicate rows of a run are dropped."""
        df = df.copy()
        df["_arm"] = df.apply(armfn, axis=1)
        df = df.dropna(subset=["_arm", metric])
        df = df.drop_duplicates(subset=["_arm", "slice_index"], keep="last")
        return {arm: dict(zip(g["slice_index"], g[metric])) for arm, g in df.groupby("_arm")}

    per_cell_ks = pd.read_csv(R("supp_per_cell_ks.csv"))

    def ks(sweep, armfn):
        return tidy(per_cell_ks[per_cell_ks.sweep == sweep], armfn, "mean_ks_cell")

    # ------------------------------------------------------------ reference guidance
    head("REFERENCE-GUIDANCE STRENGTH  (contrasts against the adopted s = 75)")

    def guidance_level(r):
        if r["expression_inference_type"] == "lookup":
            return "lookup"
        if not r["expression_weak_teacher_forcing"]:
            return "g0"
        return f"g{int(r['wtf_strength_gene'])}"

    for atlas, setting in (("MERSCOPE (rq1)", "rq1"), ("MERFISH  (rq3)", "rq3")):
        df = pd.read_csv(R(f"supp_ref_guidance_{setting}.csv"))
        f1, sp = tidy(df, guidance_level, F1), tidy(df, guidance_level, SP)
        k = ks(f"ref_guidance_{setting}", guidance_level)
        for name, tab, inv in (("soft-F1", f1, False), ("soft-Spearman", sp, False),
                               ("per-cell KS", k, True)):
            print(f"\n{atlas}  {name}   (positive = arm exceeds s=75"
                  f"{'; lower is better' if inv else ''})")
            for arm in ("g0", "g25", "g50", "g100", "lookup"):
                if arm in tab:
                    paired(tab[arm], tab["g75"], f"{arm} - g75", invert=inv)
        print(f"\n{atlas}  extra contrasts")
        for arm in ("g50", "g75", "g100", "g25"):
            paired(f1[arm], f1["lookup"], f"soft-F1: {arm} - lookup")
        paired(sp["g25"], sp["lookup"], "soft-Spearman: g25 - lookup")
        paired(k["g100"], k["g75"], "per-cell KS: g100 - g75", invert=True)

    # ------------------------------------------------------------- location guidance
    head("LOCATION GUIDANCE  (eta ladder at h = 0.01; bandwidth ladder at eta = 0.01)")
    lg = pd.read_csv(R("supp_location_guidance.csv"))
    model = lg[lg.location_inference_type == "model"]
    eta = model[np.isclose(model.kde_bandwidth, 0.01)]
    bw = model[np.isclose(model.guidance_signal, 0.01)]
    base = lg[lg.location_inference_type == "closest_slice"]

    def by(df, metric, key):
        return {round(a, 6): dict(zip(g["slice_index"], g[metric])) for a, g in df.groupby(key)}

    for r in ("0.5", "0.4", "0.3"):
        for metric in (f"gridized_l1_distance@{r} r", f"gridized_kl_divergence@{r} r"):
            t = by(eta, metric, "guidance_signal")
            print(f"\neta ladder, {metric}  (negative = better than unguided)")
            for e in (0.005, 0.01, 0.05, 0.1):
                paired(t[e], t[0.0], f"eta={e:g} - eta=0", invert=True)
    for metric in ("gridized_l1_distance@0.5 r", "gridized_kl_divergence@0.5 r"):
        t = by(bw, metric, "kde_bandwidth")
        print(f"\nbandwidth ladder at eta=0.01, {metric}  (negative = better than h=0.01)")
        for h in (0.005, 0.02, 0.05):
            paired(t[h], t[0.01], f"h={h:g} - h=0.01", invert=True)

    print("\nlocation model (unguided) against the closest-slice baseline")
    unguided = eta[eta.guidance_signal == 0.0]
    for metric in ("gridized_l1_distance@0.5 r", "gridized_kl_divergence@0.5 r"):
        u = dict(zip(unguided["slice_index"], unguided[metric]))
        b = dict(zip(base["slice_index"], base[metric]))
        paired(u, b, f"model - closest_slice, {metric}", invert=True)
        ratio = np.array([b[k] / u[k] for k in sorted(u)])
        print(f"    per-section ratio baseline/model: mean {ratio.mean():.2f} "
              f"+/- {ratio.std(ddof=1) / np.sqrt(len(ratio)):.2f}  median {np.median(ratio):.2f}  "
              f"(range {ratio.min():.2f}-{ratio.max():.2f})")

    # ------------------------------------------------------------------- decoding
    head("EXPRESSION DECODING  (contrasts against the default m = 0.01, k = 5)")

    def decoding_arm(r):
        return f"m{r['expression_select_max_index_margin']:g}_k{int(r['top_k'])}"

    df = pd.read_csv(R("supp_decoding.csv"))
    default = "m0.01_k5"
    for name, tab, inv in (("soft-F1", tidy(df, decoding_arm, F1), False),
                           ("soft-Spearman", tidy(df, decoding_arm, SP), False),
                           ("per-cell KS", ks("decoding", decoding_arm), True)):
        print(f"\n{name}   (positive = arm exceeds the default)")
        for arm in sorted(tab):
            if arm != default:
                paired(tab[arm], tab[default], f"{arm} - default", invert=inv)

    # ------------------------------------------------ cell-type sampling temperature
    head("CELL-TYPE SAMPLING TEMPERATURE  (contrasts against the adopted T = 1)")

    def temperature_arm(r):
        return f"T{r['cluster_temperature']:g}"

    df = pd.read_csv(R("supp_celltype_argmax.csv"))
    extra = [c for c in df.columns if "tvd" in c.lower() or "rare" in c.lower()]
    for metric in [F1, SP] + extra:
        tab = tidy(df, temperature_arm, metric)
        print(f"\n{metric}")
        for arm in sorted(tab):
            if arm != "T1":
                paired(tab[arm], tab["T1"], f"{arm} - T=1")
    tab = ks("argmax", temperature_arm)
    print("\nper-cell KS")
    for arm in sorted(tab):
        if arm != "T1":
            paired(tab[arm], tab["T1"], f"{arm} - T=1", invert=True)

    # -------------------------------------------------------------------- cascade
    head("ERROR PROPAGATION ACROSS STAGES  (contrasts against the full pipeline)")
    names = {("model", "model", "model"): "full pipeline",
             ("skip", "skip", "model"): "location + cell-type oracle",
             ("skip", "model", "model"): "location oracle",
             ("model", "skip", "model"): "cell-type oracle",
             ("model", "model", "lookup"): "expression -> lookup",
             ("closest_slice", "model", "model"): "location -> closest slice",
             ("model", "majority_baseline", "model"): "cell type -> majority",
             ("closest_slice", "majority_baseline", "lookup"): "all stages rule-based"}

    def cascade_arm(r):
        return names.get((r["location_inference_type"], r["cluster_inference_type"],
                          r["expression_inference_type"]))

    df = pd.read_csv(R("supp_cascade.csv"))
    f1, sp = tidy(df, cascade_arm, F1), tidy(df, cascade_arm, SP)
    k = ks("cascade", cascade_arm)
    for name, tab, inv in (("soft-F1", f1, False), ("soft-Spearman", sp, False),
                           ("per-cell KS", k, True)):
        print(f"\n{name}   (positive = arm exceeds the full pipeline)")
        for arm in tab:
            if arm != "full pipeline":
                paired(tab[arm], tab["full pipeline"], arm, invert=inv)
    print("\narm-against-arm contrasts quoted in the text")
    paired(f1["cell type -> majority"], f1["location -> closest slice"],
           "soft-F1: cell-type substitution - location substitution")
    paired(f1["cell type -> majority"], f1["expression -> lookup"],
           "soft-F1: cell-type substitution - expression substitution")
    paired(f1["location -> closest slice"], f1["expression -> lookup"],
           "soft-F1: location substitution - expression substitution")
    spread = np.array([max(k[a][s] for a in k) - min(k[a][s] for a in k)
                       for s in sorted(k["full pipeline"])])
    print(f"  per-section range of per-cell KS over all 8 configurations: "
          f"mean {spread.mean():.4f} +/- {spread.std(ddof=1) / np.sqrt(len(spread)):.4f} "
          f"(max {spread.max():.4f})")

    print("\ninteraction: is the joint oracle more than the sum of the single oracles?")
    keys = sorted(f1["full pipeline"])
    gain = lambda arm: np.array([f1[arm][s] - f1["full pipeline"][s] for s in keys])
    inter = gain("location + cell-type oracle") - gain("location oracle") - gain("cell-type oracle")
    t, p = stats.ttest_1samp(inter, 0)
    print(f"  joint - (location + cell-type) alone           {inter.mean():+.4f} +/- "
          f"{inter.std(ddof=1) / np.sqrt(len(inter)):.4f}   t={t:+7.2f}  P={p:.2e}  "
          f"{int((inter > 0).sum())}/{len(inter)} sections positive")

    # ----------------------------------------------------------------- stage-wise
    head("PER-STAGE COMPARISON  (paired by section at each reference distance d)")
    distance = {"rq3_v2_d1": 1, "rq3_v2_d2": 2, "rq3_v2_d3": 3, "rq3_v2_d4": 4}
    spatialz = pd.read_csv(args.spatialz) if args.spatialz else None

    def stage_table(df, metric, method_col=None):
        df = df.copy()
        if method_col:
            df["_m"] = df[method_col]
        else:
            for col, mp in (("expression_inference_type", {"model": "MIMYR", "lookup": "Lookup"}),
                            ("cluster_inference_type", {"model": "MIMYR", "majority_baseline": "Lookup"}),
                            ("location_inference_type", {"model": "MIMYR", "closest_slice": "Lookup"})):
                if df[col].map(mp).notna().all():
                    df["_m"] = df[col].map(mp)
                    break
        df["_d"] = df["data_mode"].map(distance)
        df = df.dropna(subset=["_d", metric, "_m"])
        df = df.drop_duplicates(subset=["_m", "_d", "slice_index"], keep="last")
        return {(m, int(d)): dict(zip(g["slice_index"], g[metric]))
                for (m, d), g in df.groupby(["_m", "_d"])}

    def stage(title, ours, metric, method_col=None, spatialz_metric=None, invert=False):
        print(f"\n{title}")
        tab = stage_table(pd.read_csv(R(ours)), metric, method_col)
        if spatialz is not None and spatialz_metric:
            tab.update(stage_table(spatialz, spatialz_metric, "method"))
        for other in ("Lookup", "SpatialZ"):
            pooled, per_section = [], {}
            for d in (1, 2, 3, 4):
                if ("MIMYR", d) in tab and (other, d) in tab:
                    a, b = tab[("MIMYR", d)], tab[(other, d)]
                    paired(a, b, f"MIMYR - {other}, d={d}", invert=invert)
                    for s in sorted(set(a) & set(b)):
                        pooled.append(a[s] - b[s])
                        per_section.setdefault(s, []).append(a[s] - b[s])
            if pooled:
                v = np.array(pooled)
                npos = int((v > 0).sum())
                print(f"  {'(descriptive) all 20 section-distance pairs':<46s} {v.mean():+.4f}"
                      f"                    {len(v) - npos if invert else npos}/{len(v)} pairs improve")
                # the same five sections recur at every d, so test the per-section
                # mean over d (n = 5) rather than the 20 dependent pairs
                v = np.array([np.mean(x) for x in per_section.values()])
                t, p = stats.ttest_1samp(v, 0)
                npos = int((v > 0).sum())
                print(f"  {'section-level mean over d (n=5)':<46s} {v.mean():+.4f} +/- "
                      f"{v.std(ddof=1) / np.sqrt(len(v)):.4f}   t={t:+7.2f}  P={p:.2e}  "
                      f"{len(v) - npos if invert else npos}/{len(v)} sections improve")

    stage("LOCATION, binned L1 @ 0.5 (lower is better)", "supp_stage_location.csv",
          "gridized_l1_distance@0.5 r", spatialz_metric="gridized_l1_distance@0.5 r", invert=True)
    stage("CELL TYPE, exact class-level accuracy (higher is better)",
          "supp_stage_celltype_class.csv", "exact_accuracy_class", "method",
          spatialz_metric="exact_accuracy_class")
    stage("CELL TYPE, soft accuracy @ 0.05 (higher is better)",
          "supp_stage_celltype_class.csv", "soft_accuracy@0.05", "method",
          spatialz_metric="soft_accuracy@0.05")
    stage("CELL TYPE, exact token-level accuracy (MIMYR against lookup only)",
          "supp_stage_celltype_class.csv", "exact_accuracy_token", "method")
    stage("EXPRESSION, soft-F1 @ 0.05 (higher is better)", "supp_stage_expression.csv", F1,
          spatialz_metric="stage3_" + F1)
    stage("EXPRESSION, soft-Spearman @ 0.05 (higher is better)", "supp_stage_expression.csv", SP,
          spatialz_metric="stage3_" + SP)


if __name__ == "__main__":
    main()
