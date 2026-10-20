"""Publish MIMYR's pretrained checkpoints and benchmark data to the Hugging Face Hub.

Maintainer script. Uploads to the repos named in `mimyr.pretrained`
(MODEL_REPO, DATA_REPO), creating them private unless --public is given, so a
release can be checked before anyone else can see it.

    python scripts/publish_to_hub.py models                 # checkpoints + model card
    python scripts/publish_to_hub.py data --archives_dir D  # archives + dataset card
"""

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

from mimyr import pretrained

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Hub path -> source checkpoint on the lab cluster (the files behind the paper's figures).
CHECKPOINT_SOURCES = {
    "location/smoothtune_conditional_ddpm_2d_checkpoint_400.pt":
        f"{REPO_ROOT}/model_checkpoints/smoothtune_conditional_ddpm_2d_checkpoint_400.pt",
    "location/location_zhuang1_ddpm_2d_checkpoint_400.pt":
        f"{REPO_ROOT}/model_checkpoints/location_zhuang1_ddpm_2d_checkpoint_400.pt",
    "celltype/best_model_rq1.pt": f"{REPO_ROOT}/model_checkpoints/best_model_rq1.pt",
    "celltype/celltype_zhuang1.pt": f"{REPO_ROOT}/model_checkpoints/celltype_zhuang1.pt",
    "expression/merscope_epoch165_model.pt":
        "/compute/oven-0-13/skrieger/TG-base-noreweight-slaf-binss-lambda1/epoch165_model.pt",
    "expression/zhuang1_epoch500_model.pt":
        "/compute/oven-0-13/skrieger/TG-base-zhuang1-slaf/epoch500_model.pt",
    "expression/zhuang2-finetuned_epoch502_model.pt":
        "/compute/oven-0-13/skrieger/FT-rq3_v2-binss-lambda1-omitx/epoch502_model.pt",
    "expression/zhuang3-finetuned_epoch510_model.pt":
        "/compute/oven-0-13/skrieger/FT-rq4-binss-lambda1-omitx/epoch510_model.pt",
}


def stage_models(staging):
    referenced = {p for preset in pretrained.PRESETS.values() for p in preset["checkpoints"].values()}
    missing = referenced - set(CHECKPOINT_SOURCES)
    if missing:
        raise RuntimeError(f"presets reference checkpoints with no source: {sorted(missing)}")
    for hub_path, src in CHECKPOINT_SOURCES.items():
        dst = os.path.join(staging, hub_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)
    shutil.copy(os.path.join(REPO_ROOT, "hub", "model_card.md"), os.path.join(staging, "README.md"))


def stage_data(staging, archives_dir, archives):
    for name in archives:
        os.symlink(os.path.join(archives_dir, name), os.path.join(staging, name))
    shutil.copy(os.path.join(REPO_ROOT, "hub", "dataset_card.md"), os.path.join(staging, "README.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["models", "data"])
    parser.add_argument("--archives_dir", help="Folder with the archives from build_hub_archives.py")
    parser.add_argument("--archives", nargs="+",
                        default=list(pretrained.DATA_ARCHIVES),
                        help="Which data archives to upload (default: all)")
    parser.add_argument("--public", action="store_true", help="Create the repo public (default: private)")
    args = parser.parse_args()

    api = HfApi()
    if args.what == "models":
        repo_id, repo_type = pretrained.MODEL_REPO, "model"
    else:
        repo_id, repo_type = pretrained.DATA_REPO, "dataset"
    api.create_repo(repo_id, repo_type=repo_type, private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        if args.what == "models":
            stage_models(staging)
        else:
            if not args.archives_dir:
                parser.error("data needs --archives_dir")
            stage_data(staging, args.archives_dir, args.archives)
        api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=staging,
                          commit_message=f"Upload MIMYR {args.what}")
    print(f"Uploaded to https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}")


if __name__ == "__main__":
    main()
