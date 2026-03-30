import argparse
from dataclasses import dataclass
from datetime import datetime
import zipfile

import gdown
import yaml

from models.combined_model import CombinedModel
from data_loader import SliceDataLoader
from models.biological_model import KDEModelForGuidance
import torch
import numpy as np
from inference import Inference
from evaluator import Evaluator
import copy, os, pandas as pd
import pickle as pkl

import json
import gc


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@dataclass
class TrainConfig:
    degree: int = 7
    hidden_sizes: tuple = (1024, 2048, 4096, 2048, 1024)
    activation: str = "silu"
    batchnorm: bool = False
    dropout: float = 0.0
    feature_type: str = "poly"
    num_rff_features: int = 256
    rff_gamma: float = 100.0
    rff_seed: int | None = None
    n_timesteps: int = 70
    schedule_type: str = "cosine"
    beta_start: float = 1e-10
    beta_end: float = 1e-9
    cosine_s: float = 0.008
    batch_size: int = 4096 * 50
    lr: float = 2e-4
    weight_decay: float = 0
    epochs: int = 1000000
    grad_clip: float = None
    ema_decay: float = 0.999

# ----------------- argparse -----------------
def get_args():
    parser = argparse.ArgumentParser(description="Run inference with config options.")

    # config file (loaded first so its values become defaults; explicit CLI flags override them)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. Values here become defaults; CLI flags take precedence.",
    )

    # data / model paths
    parser.add_argument(
        "--run_mode", type=str, default="inference", help="Mode for SliceDataLoader", choices=["train", "inference"]
    )
    parser.add_argument(
        "--training_slice_directory", type=str, default="data/cleaned_versions", help="Training adata, if mode is set to train"
    )
    parser.add_argument(
        "--val_slice_directory", type=str, default="data/cleaned_versions", help="Validation adata, if mode is set to train"
    )
    parser.add_argument(
        "--skip_combined_fit", action="store_true",
        help="Skip combined_model.fit() (location/celltype training) and go straight to expression model training",
    )


    parser.add_argument(
        "--data_mode", type=str, default="rq1", help="Mode for SliceDataLoader"
    )
    parser.add_argument(
        "--data_label",
        type=str,
        default="cluster",
        help="Label type for SliceDataLoader",
    )
    parser.add_argument(
        "--location_model_checkpoint",
        type=str,
        default="model_checkpoints/smoothtune_conditional_ddpm_2d_checkpoint_400.pt",
        help="Path to trained location checkpoint",
    )
    parser.add_argument(
        "--cluster_model_checkpoint",
        type=str,  
        default="model_checkpoints/best_model_rq1.pt",
        help="Path to trained CelltypeModel checkpoint",
    )
    parser.add_argument(
        "--expression_model_checkpoint",
        type=str,  ### CHANGE
        default="model_checkpoints/TG-base4_epoch4_model.pt",
        help="Path to trained expression checkpoint",
    )

    parser.add_argument(
        "--location_inference_type",
        type=str,
        default="skip",
        help="Type of location inference",
    )
    parser.add_argument(
        "--kde_bandwidth", type=float, default=0.01, help="Bandwidth for KDE"
    )

    parser.add_argument(
        "--cluster_inference_type",
        type=str,
        default="model",
        help="How to infer subclass",
    )
    parser.add_argument(
        "--expression_inference_type",
        type=str,
        default="end",
        help="How to infer gene expression",
    )

    # training hyperparams
    parser.add_argument(
        "--epochs", type=int, default=500, help="Training epochs for CelltypeModel"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for CelltypeModel",
    )
    parser.add_argument(
        "--guidance_signal",
        type=float,
        default=0.01,
        help="Guidance signal for classifier-based guidance",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for CelltypeModel"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cpu/cuda)"
    )

    parser.add_argument(
        "--metrics",
        type=lambda s: s.split(","),
        help="Comma-separated list of metrics to compute (soft_accuracy,soft_correlation,neighborhood_enrichment,soft_precision)",
        default=[
            "soft_accuracy"
        ],  # ,"soft_accuracy", "soft_correlation", "neighborhood_enrichment", "soft_precision"],
    )
    parser.add_argument(
        "--metric_sampling",
        type=int,
        default=1,
        help="Percentage of samples to use for metric computation",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/output.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--meta_info",
        type=str,
        default="4hierarchy_metainfo_mouse_geneunion2_DAG.pt",
        help="meta_info file path for GE prediction",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the rq1 data (quantized slices)",
    )
    parser.add_argument(
        "--zhuang_data_dir",
        type=str,
        default=None,
        help="Base directory for Zhuang MERFISH data; must contain Zhuang-ABCA-2 and Zhuang-ABCA-3 subdirs (required for rq2, rq3, rq4 modes)",
    )
    parser.add_argument(
        "--diseased_data_dir",
        type=str,
        default=None,
        help="Directory containing diseased (5xFAD Trem2) slices (required for rq5 mode)",
    )
    parser.add_argument(
        "--rq3_v2_rq1_test_indices",
        type=str,
        default=None,
        help="Comma-separated Zhuang-ABCA-2 test slice indices for rq3_v2_rq1 mode (default: '1,10,20,30,40')",
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="artifacts",
        help="Directory to save artifacts",
    )

    # Expression model training args (used when --run_mode train)
    parser.add_argument(
        "--expression_output_dir",
        type=str,
        default="model_checkpoints/expression_finetuned",
        help="Directory to save the finetuned expression model",
    )
    parser.add_argument(
        "--expression_epochs", type=int, default=5,
        help="Number of fine-tuning epochs for the expression model",
    )
    parser.add_argument(
        "--expression_batch_size", type=int, default=8,
        help="Batch size for expression model training",
    )
    parser.add_argument(
        "--expression_lr", type=float, default=5e-5,
        help="Learning rate for expression model training",
    )
    parser.add_argument(
        "--expression_lambda_val", type=float, default=1.0,
        help="Weight on the expression MSE loss term",
    )
    parser.add_argument(
        "--expression_max_len", type=int, default=512,
        help="Max sequence length for prompts + genes",
    )
    parser.add_argument(
        "--expression_save_frequency", type=int, default=10,
        help="Number of epochs between expression model checkpoints",
    )
    parser.add_argument(
        "--expression_epoch_samples", type=int, default=-1,
        help="Rows to draw per epoch for expression training (-1 uses full dataset)",
    )
    parser.add_argument(
        "--expression_log_per_steps", type=int, default=100,
        help="Log to W&B every this many steps during expression training",
    )
    parser.add_argument(
        "--expression_new_expression_size", type=int, default=None,
        help="Override n_expression_level in the expression model",
    )
    parser.add_argument(
        "--expression_no_shuffle", action="store_true",
        help="Disable DataLoader shuffling for expression training",
    )
    parser.add_argument(
        "--expression_xyz_noise", action="store_true",
        help="Add noise to x,y,z coordinates during expression model training",
    )
    parser.add_argument(
        "--expression_adata2", type=str, default=None,
        help="Optional path to a second .h5ad file (e.g. scRNA-seq) to merge into training",
    )
    parser.add_argument(
        "--expression_metadata_dir", type=str,
        default="model_checkpoints/metadata",
        help="Directory containing metadata files used by the expression model (edges_x/y/z.pkl, hierarchy.pkl, meta_info .pt)",
    )
    parser.add_argument(
        "--expression_from_finetuned", action="store_true",
        help="Indicate that the expression checkpoint is already finetuned (affects weight loading)",
    )
    parser.add_argument(
        "--expression_model_size", type=str, default=None,
        choices=["small", "medium", "large"],
        help="Model size to initialise from scratch if no expression checkpoint is provided",
    )
    parser.add_argument(
        "--expression_rebalance_only", action="store_true",
        help="Ignore CSV weights; use uniform per-cell weights then rebalance mass across st/scrna groups",
    )

    # Two-pass: extract --config first, load YAML, set as new defaults, then re-parse
    # so that explicit CLI flags still take precedence over the config file.
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        with open(pre_args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        # Skip null values so they don't clobber argparse defaults
        file_cfg = {k: v for k, v in file_cfg.items() if v is not None}
        parser.set_defaults(**file_cfg)

    return parser.parse_args()


# ----------------- util -----------------
def write_row(row, path):
    header = not os.path.isfile(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def already_done(cfg, path):
    if not os.path.isfile(path):
        return False
    existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
    check_cols = [k for k in cfg.keys() if k in existing_cols]
    df = pd.read_csv(path, usecols=check_cols)
    return any((df == pd.Series({k: cfg[k] for k in check_cols})).all(axis=1))


# ----------------- main -----------------


def main():
    args = get_args()
    cfg = args.__dict__


    if not os.path.exists(cfg["artifact_dir"]):
        os.makedirs(cfg["artifact_dir"], exist_ok=True)

    if not os.path.exists(cfg["data_dir"]):
        gdown.download(id="1iJX3z9S_biGCdpc-uQWwJyFImhv2mdU8", output="data.zip")
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    
    if not os.path.exists("model_checkpoints"):
        gdown.download(id="1OSh5JfXg2OXVfTIyGR33PkvQYGNqFXLD", output="model_checkpoints.zip")
        with zipfile.ZipFile("model_checkpoints.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    print(cfg)
    slice_data_loader = SliceDataLoader(
        mode=args.data_mode,
        label=args.data_label,
        cfg=copy.deepcopy(cfg),
        metadata_dir=args.expression_metadata_dir,
    )

    cfg["full_gene_panel"] = True

    if args.run_mode != "train":
        slice_data_loader.prepare()
        temp_test_slices = slice_data_loader.test_slices.copy()
        temp_ref_slices = slice_data_loader.reference_slices.copy()

    artifact_dir = cfg['artifact_dir']

    combined_model = CombinedModel(
        location_model_checkpoint=args.location_model_checkpoint,
        celltype_model_checkpoint=args.cluster_model_checkpoint,
        gene_exp_model_checkpoint=args.expression_model_checkpoint,
    )

    if args.run_mode == "train":
        import torch.distributed as _dist
        os.environ["NCCL_P2P_DISABLE"] = "1"
        if _dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
            _dist.init_process_group(backend="nccl")
            _rank = _dist.get_rank()
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        else:
            _rank = 0

        if _rank == 0 and not args.skip_combined_fit:
            combined_model.fit(args.training_slice_directory, args.val_slice_directory, cfg)

        if _dist.is_available() and _dist.is_initialized():
            _dist.barrier()

        # Train the gene expression module via finetune_mimyr
        from models.generative_transformer import train_expression_model as _train_expression_model
        import argparse as _argparse

        expr_args = _argparse.Namespace(
            # paths / identity
            ckp_path=args.expression_model_checkpoint,
            meta_info=os.path.join(args.data_dir, args.meta_info),
            output_dir=args.expression_output_dir,
            # data
            data_mode=args.data_mode,
            data_label=args.data_label,
            data_dir=args.data_dir,
            zhuang_data_dir=args.zhuang_data_dir,
            adata2=args.expression_adata2,
            metadata_dir=args.expression_metadata_dir,
            # training hyperparams
            epochs=args.expression_epochs,
            batch_size=args.expression_batch_size,
            lr=args.expression_lr,
            device=args.device,
            lambda_val=args.expression_lambda_val,
            max_len=args.expression_max_len,
            no_shuffle=args.expression_no_shuffle,
            num_workers=4,
            save_frequency=args.expression_save_frequency,
            xyz_noise=args.expression_xyz_noise,
            epoch_samples=args.expression_epoch_samples,
            seed=42,
            log_per_steps=args.expression_log_per_steps,
            bin_edges_file=None,
            # model init
            from_finetuned=args.expression_from_finetuned,
            model_size=args.expression_model_size,
            overwrite_vocab_size=None,
            new_expression_size=args.expression_new_expression_size,
            # sampling
            disable_sampling_probs=True,
            use_sampling_probs=False,
            sampling_col="sampling_prob",
            # misc
            dummy=False,
            kv_cache=False,
            val_split=0.0,  # data is pre-split by SliceDataLoader; this is informational only
            rebalance_only=args.expression_rebalance_only,
        )
        _train_expression_model(expr_args)
        exit(0)


    for i, slice in enumerate(temp_test_slices):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["artifact_dir"] = f"{artifact_dir}/{timestamp}"
        os.makedirs(cfg["artifact_dir"], exist_ok=True)

        slice_data_loader.test_slices = [slice]

        slice_data_loader.reference_slices = temp_ref_slices[
            2 * i : 2 * i + 2
        ]
        if len(slice_data_loader.reference_slices) == 0:
            slice_data_loader.reference_slices = temp_ref_slices[-2:]

        cfg["slice_index"] = i


        closest_ref_slice = np.argsort(
            [
                np.square(
                    ref_slice.obsm["aligned_spatial"].mean(0)[-1]
                    - slice_data_loader.test_slices[0]
                    .obsm["aligned_spatial"]
                    .mean(0)[-1]
                )
                for ref_slice in slice_data_loader.reference_slices
            ]
        )[1]
        best_ref_slice = slice_data_loader.reference_slices[closest_ref_slice].copy()
        if args.data_mode == "rq4":
            best_ref_slice.obsm["aligned_spatial"][:, 0] = (
                slice_data_loader.test_slices[0].obsm["aligned_spatial"][:, 0].mean(0)
            )
        else:
            best_ref_slice.obsm["aligned_spatial"] = best_ref_slice.obsm[
                "aligned_spatial"
            ][:, :2]


        kdemodel = KDEModelForGuidance(
            [best_ref_slice], bandwidth=args.kde_bandwidth
        )
        kdemodel.fit()


        if already_done(cfg, args.out_csv):
            print("skip", cfg)
            return

        inf = Inference(
            combined_model, 
            kdemodel,
            slice_data_loader,
            copy.deepcopy(cfg),
        )

        pred = inf.run_inference(slice_data_loader.test_slices)
        print("Sending pred to evaluator...", pred)
        res = Evaluator(cfg, metadata_dir=args.expression_metadata_dir).evaluate(
            pred, slice_data_loader.test_slices[0], sample=args.metric_sampling
        )
        res = {k: float(v) for k, v in res.items()}

        row = {**cfg, **res}
        write_row(row, args.out_csv)
        print("wrote", cfg)
        # save config to artifact dir as json

        # save config + results into artifact dir
        cfg_path = os.path.join(cfg["artifact_dir"], "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        res_path = os.path.join(cfg["artifact_dir"], "results.json")
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

        # optionally also save predictions
        pred_path = os.path.join(cfg["artifact_dir"], "pred.pkl")
        with open(pred_path, "wb") as f:
            pkl.dump(pred, f)

        # delete all local variables and collect garbage
        del pred, inf
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
