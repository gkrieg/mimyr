"""Build the per-dataset archives published to the MIMYR Hugging Face dataset repo.

Maintainer script. Each archive holds one dataset's files with paths relative to
`--data_dir`, so `mimyr.pretrained.ensure_data` can extract it there directly:

    rq1-merscope.zip    subclass_z1_d338_0_rotated/sec_*.h5ad
    Zhuang-ABCA-<n>.zip Zhuang-ABCA-<n>/*  and  ccf-ABCA-<n>.csv
    Trem2-5xFAD.zip     cleaned_versions/*

The file list is the one previously distributed as data.zip on Google Drive
(`--members`, a JSON of {relative path: size}); every file is checked against
that size before it is added, so the archives are byte-identical to it.

Usage:
    python scripts/build_hub_archives.py --members data_members.json --out_dir staging/data
"""

import argparse
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

# Where each top-level entry lives on the lab cluster.
SOURCES = {
    "subclass_z1_d338_0_rotated": "/work/magroup/skrieger/tissue_generator/quantized_slices",
    "Zhuang-ABCA-": "/work/magroup/skrieger/MERFISH_BICCN/processed_data",
    "ccf-ABCA-": "/work/magroup/skrieger/tissue_generator/quantized_slices",
    "cleaned_versions": "/work/magroup/skrieger/tissue_generator/CCF_registration/ccf_aligned_Trem2_5xFAD",
}


def archive_for(member):
    top = member.split("/")[0]
    if top == "subclass_z1_d338_0_rotated":
        return "rq1-merscope.zip"
    if top.startswith("Zhuang-ABCA-"):
        return f"{top}.zip"
    if top.startswith("ccf-ABCA-"):
        return f"Zhuang-ABCA-{top[len('ccf-ABCA-'):-len('.csv')]}.zip"
    if top == "cleaned_versions":
        return "Trem2-5xFAD.zip"
    raise ValueError(f"no archive for {member}")


def source_for(member):
    for prefix, root in SOURCES.items():
        if member.startswith(prefix):
            return os.path.join(root, member)
    raise ValueError(f"no source for {member}")


def build(name, members, out_dir):
    out = os.path.join(out_dir, name)
    tmp = out + ".partial"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for member, size in sorted(members.items()):
            src = source_for(member)
            actual = os.path.getsize(src)
            if actual != size:
                raise RuntimeError(f"{src}: size {actual} != published {size}")
            z.write(src, arcname=member)
    os.replace(tmp, out)
    return f"{name}: {len(members)} files, {os.path.getsize(out) / 1e9:.2f} GB"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--members", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    members = json.load(open(args.members))
    groups = {}
    for member, size in members.items():
        groups.setdefault(archive_for(member), {})[member] = size
    os.makedirs(args.out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=len(groups)) as pool:
        for line in pool.map(build, groups, groups.values(), [args.out_dir] * len(groups)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
