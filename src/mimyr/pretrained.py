"""Pretrained MIMYR models and benchmark data on the Hugging Face Hub.

Checkpoints live in the model repo ``gkrieg13/mimyr`` and the benchmark sections
in the dataset repo ``gkrieg13/mimyr-data``. Files are downloaded on first use and
cached by ``huggingface_hub`` (``~/.cache/huggingface`` by default; set ``HF_HOME``
to move it).

Each preset is one of the models reported in the paper: its three checkpoints plus
the inference settings that produced the published numbers.
"""

import os
import shutil
import tempfile
import zipfile

MODEL_REPO = "gkrieg13/mimyr"
DATA_REPO = "gkrieg13/mimyr-data"
# Branch, tag, or commit to download from. Pin it to a release tag for reproducibility.
REVISION = os.environ.get("MIMYR_HUB_REVISION", "main")

_LOC_MERSCOPE = "location/smoothtune_conditional_ddpm_2d_checkpoint_400.pt"
_LOC_ZHUANG1 = "location/location_zhuang1_ddpm_2d_checkpoint_400.pt"
_CT_MERSCOPE = "celltype/best_model_rq1.pt"
_CT_ZHUANG1 = "celltype/celltype_zhuang1.pt"

PRESETS = {
    "merscope": {
        "description": "Base model trained on the MERSCOPE whole-brain atlas (paper Fig. 2; data_mode rq1 or demo)",
        "orientation": "coronal",
        "checkpoints": {
            "location": _LOC_MERSCOPE,
            "celltype": _CT_MERSCOPE,
            "expression": "expression/merscope_epoch165_model.pt",
        },
        "settings": {
            "expression_weak_teacher_forcing": True,
            "wtf_strength_gene": 75.0,
            "wtf_strength_bin": 75.0,
            "wtf_strength_value": 75.0,
            "metric_filter_by_gt": True,
        },
    },
    "zhuang1": {
        "description": "Base model trained on Zhuang-ABCA-1, applied zero-shot to Zhuang-ABCA-2 (paper Fig. 3; data_mode rq3_v2)",
        "orientation": "coronal",
        "checkpoints": {
            "location": _LOC_ZHUANG1,
            "celltype": _CT_ZHUANG1,
            "expression": "expression/zhuang1_epoch500_model.pt",
        },
        "settings": {
            "expression_weak_teacher_forcing": True,
            "wtf_strength_gene": 75.0,
            "wtf_strength_bin": 75.0,
            "wtf_strength_value": 75.0,
            "metric_filter_by_gt": True,
            "omit_x": True,
        },
    },
    "zhuang2-finetuned": {
        "description": "Zhuang-1 base model fine-tuned on Zhuang-ABCA-2 (paper Fig. 3; data_mode rq3_v2)",
        "orientation": "coronal",
        "checkpoints": {
            "location": _LOC_ZHUANG1,
            "celltype": _CT_ZHUANG1,
            "expression": "expression/zhuang2-finetuned_epoch502_model.pt",
        },
        "settings": {
            "expression_weak_teacher_forcing": True,
            "wtf_strength_gene": 75.0,
            "wtf_strength_bin": 75.0,
            "wtf_strength_value": 75.0,
            "metric_filter_by_gt": True,
            "omit_x": True,
        },
    },
    "zhuang3-finetuned": {
        "description": "Model fine-tuned on the sagittal Zhuang-ABCA-3 brain (paper Fig. 4; data_mode rq4)",
        "orientation": "sagittal",
        "checkpoints": {
            "location": _LOC_MERSCOPE,
            "celltype": _CT_MERSCOPE,
            "expression": "expression/zhuang3-finetuned_epoch510_model.pt",
        },
        "settings": {
            "expression_weak_teacher_forcing": True,
            "wtf_strength_gene": 50.0,
            "wtf_strength_bin": 50.0,
            "wtf_strength_value": 50.0,
            "metric_filter_by_gt": True,
        },
    },
}

DEFAULT_PRESET = "merscope"

# Archives in the dataset repo. Each extracts into --data_dir; `folder` is what it
# creates, so an archive is only downloaded when that folder is missing.
DATA_ARCHIVES = {
    "rq1-merscope.zip": "subclass_z1_d338_0_rotated",
    "Zhuang-ABCA-1.zip": "Zhuang-ABCA-1",
    "Zhuang-ABCA-2.zip": "Zhuang-ABCA-2",
    "Zhuang-ABCA-3.zip": "Zhuang-ABCA-3",
    "Zhuang-ABCA-4.zip": "Zhuang-ABCA-4",
}


def checkpoint(preset, role):
    """Local path to a preset's checkpoint for `role` (location, celltype, expression),
    downloading it from the Hub on first use."""
    if preset not in PRESETS:
        raise ValueError(f"unknown pretrained model {preset!r}; choose from {', '.join(PRESETS)}")
    return _download(MODEL_REPO, "model", PRESETS[preset]["checkpoints"][role],
                     hint=f"pass the {role} checkpoint path explicitly instead")


def _download(repo_id, repo_type, filename, hint):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        return hf_hub_download(repo_id, filename, repo_type=repo_type, revision=REVISION)
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        raise FileNotFoundError(
            f"{filename} is not available from the {repo_type} repo {repo_id} "
            f"(revision {REVISION}); {hint}."
        ) from e


def archives_for(data_mode, use_rq1_train=False):
    """The data archives a data mode reads."""
    mode = str(data_mode)
    if mode == "demo":
        return []
    if mode == "rq1":
        return ["rq1-merscope.zip"]
    if mode == "base_zhuang1":
        return ["Zhuang-ABCA-1.zip"]
    if mode in ("rq2", "rq2_v2"):
        return ["Zhuang-ABCA-2.zip"] + (["rq1-merscope.zip"] if mode == "rq2_v2" else [])
    if mode.startswith("rq3"):
        return ["Zhuang-ABCA-2.zip"] + (["rq1-merscope.zip"] if use_rq1_train else [])
    if mode == "rq4":
        return ["Zhuang-ABCA-3.zip"]
    if mode == "rq4_rq3":
        return ["Zhuang-ABCA-3.zip", "Zhuang-ABCA-2.zip"]
    if mode == "rq4_noref":
        return ["Zhuang-ABCA-3.zip", "Zhuang-ABCA-1.zip"]
    return []


def ensure_data(data_mode, data_dir, use_rq1_train=False):
    """Download and extract into `data_dir` whatever `data_mode` needs and is missing."""
    from filelock import FileLock

    needed = [n for n in archives_for(data_mode, use_rq1_train)
              if not os.path.isdir(os.path.join(data_dir, DATA_ARCHIVES[n]))]
    if not needed:
        return
    os.makedirs(data_dir, exist_ok=True)
    # Several torchrun ranks may get here at once: one extracts, the rest wait.
    with FileLock(os.path.join(data_dir, ".mimyr_download.lock")):
        for name in needed:
            if not os.path.isdir(os.path.join(data_dir, DATA_ARCHIVES[name])):
                _download_archive(name, data_dir)


def _download_archive(name, data_dir):
    print(f"Downloading {name} from huggingface.co/datasets/{DATA_REPO} ...", flush=True)
    path = _download(DATA_REPO, "dataset", name,
                     hint=f"put its {DATA_ARCHIVES[name]}/ folder in {data_dir} yourself")
    # Extract to a scratch folder first and move the entries into place only
    # once complete, so an interrupted run is retried rather than half-read.
    os.makedirs(data_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{name}.", dir=data_dir)
    try:
        with zipfile.ZipFile(path) as z:
            z.extractall(staging)
        for entry in os.listdir(staging):
            target = os.path.join(data_dir, entry)
            if os.path.isdir(target):
                shutil.rmtree(target)
            os.replace(os.path.join(staging, entry), target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"Extracted {name} into {data_dir}", flush=True)
