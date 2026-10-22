"""
Write the gene name overlap between one rq1 slice and one rq3 slice to a file,
one gene per line. The result can be passed to --metric_gene_set_file at evaluation.

Usage:
    python write_gene_overlap.py \
        --data_dir data \
        --zhuang_data_dir /path/to/zhuang \
        --output gene_overlap.txt \
        [--rq1_index 0] \
        [--rq3_index 0]
"""

import argparse
import os
import scanpy as sc


def main():
    parser = argparse.ArgumentParser(description="Write rq1/rq3 gene overlap to a file.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing rq1 data (subclass_z1_d338_0_rotated subdir)")
    parser.add_argument("--zhuang_data_dir", type=str, required=True,
                        help="Base directory for Zhuang MERFISH data (must contain Zhuang-ABCA-2 subdir)")
    parser.add_argument("--output", type=str, default="gene_overlap.txt",
                        help="Output file path (one gene per line)")
    parser.add_argument("--rq1_index", type=int, default=0,
                        help="Index into the sorted rq1 slice list to load (default: 0)")
    parser.add_argument("--rq3_index", type=int, default=0,
                        help="Index into the sorted rq3 slice list to load (default: 0)")
    args = parser.parse_args()

    # --- rq1 slice ---
    rq1_dir = os.path.join(args.data_dir, "subclass_z1_d338_0_rotated")
    rq1_files = [
        "sec_05.h5ad", "sec_06.h5ad", "sec_08.h5ad", "sec_09.h5ad",
        "sec_10.h5ad", "sec_11.h5ad", "sec_12.h5ad", "sec_13.h5ad",
        "sec_14.h5ad", "sec_15.h5ad", "sec_16.h5ad", "sec_17.h5ad",
        "sec_18.h5ad", "sec_19.h5ad", "sec_24.h5ad", "sec_25.h5ad",
        "sec_26.h5ad", "sec_27.h5ad", "sec_28.h5ad", "sec_29.h5ad",
        "sec_30.h5ad", "sec_31.h5ad", "sec_32.h5ad", "sec_33.h5ad",
        "sec_35.h5ad", "sec_36.h5ad", "sec_37.h5ad", "sec_38.h5ad",
        "sec_39.h5ad", "sec_40.h5ad", "sec_42.h5ad", "sec_43.h5ad",
        "sec_44.h5ad", "sec_45.h5ad", "sec_46.h5ad", "sec_47.h5ad",
        "sec_48.h5ad", "sec_49.h5ad", "sec_50.h5ad", "sec_51.h5ad",
        "sec_52.h5ad", "sec_54.h5ad", "sec_55.h5ad", "sec_56.h5ad",
        "sec_57.h5ad", "sec_58.h5ad", "sec_59.h5ad", "sec_60.h5ad",
        "sec_61.h5ad", "sec_62.h5ad", "sec_64.h5ad", "sec_66.h5ad",
        "sec_67.h5ad",
    ]
    rq1_fname = rq1_files[args.rq1_index]
    rq1_path = os.path.join(rq1_dir, rq1_fname)
    print(f"Loading rq1 slice: {rq1_path}")
    rq1 = sc.read_h5ad(rq1_path, backed="r")
    rq1_genes = set(rq1.var_names)
    rq1.file.close()
    print(f"  {len(rq1_genes)} genes")

    # --- rq3 slice (Zhuang-ABCA-2) ---
    rq3_dir = os.path.join(args.zhuang_data_dir, "Zhuang-ABCA-2")
    rq3_files = sorted(f for f in os.listdir(rq3_dir) if f.endswith(".h5ad"))[2:-2]
    rq3_fname = rq3_files[args.rq3_index]
    rq3_path = os.path.join(rq3_dir, rq3_fname)
    print(f"Loading rq3 slice: {rq3_path}")
    rq3 = sc.read_h5ad(rq3_path, backed="r")
    rq3_genes = set(rq3.var_names)
    rq3.file.close()
    print(f"  {len(rq3_genes)} genes")

    # --- overlap ---
    overlap = sorted(rq1_genes & rq3_genes)
    print(f"Gene overlap: {len(overlap)} genes")

    with open(args.output, "w") as f:
        for gene in overlap:
            f.write(gene + "\n")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
