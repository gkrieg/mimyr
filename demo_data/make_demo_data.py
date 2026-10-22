"""Build the small demo dataset used by `python main.py` (data_mode "demo").

The demo is the first rq1 test section (sec_14) and its two neighbouring
reference sections (sec_13, sec_15), each randomly subsampled to the same
fraction of its cells. All obs columns and the full gene panel are kept, so the
demo goes through exactly the same preprocessing as rq1; only the obsm
embeddings and uns, which the pipeline never reads, are dropped.

Usage:
    python demo_data/make_demo_data.py --rq1_dir data/subclass_z1_d338_0_rotated
"""

import argparse
import os

import anndata as ad
import numpy as np

SECTIONS = ["sec_13", "sec_14", "sec_15"]  # reference, test, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rq1_dir", required=True, help="Directory with the rq1 sec_*.h5ad files")
    parser.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for name in SECTIONS:
        adata = ad.read_h5ad(os.path.join(args.rq1_dir, f"{name}.h5ad"))
        n = int(round(adata.n_obs * args.fraction))
        keep = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
        demo = ad.AnnData(X=adata.X[keep], obs=adata.obs.iloc[keep].copy(), var=adata.var.copy())
        out = os.path.join(args.out_dir, f"{name}.h5ad")
        demo.write_h5ad(out, compression="gzip")
        print(f"{name}: kept {n} of {adata.n_obs} cells -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
