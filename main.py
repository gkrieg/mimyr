import argparse
from dataclasses import dataclass
from datetime import datetime
from email import parser
import sys
import anndata as ad
from diffusion_model import DDPMTrainer
sys.path.append("generative_transformer")

from celltype_model import CelltypeModel, SkeletonCelltypeModel
from data_loader import SliceDataLoader
from biological_model import BiologicalModel2
import torch
import numpy as np
from inference import Inferernce
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
    # hidden_sizes: tuple = (512, 512)
    activation: str = "relu"
    batchnorm: bool = False
    dropout: float = 0.0
    feature_type: str = "poly"
    num_rff_features: int = 256
    rff_gamma: float = 100.0
    rff_seed: int | None = None
    n_timesteps: int = 1000
    schedule_type: str = "cosine"
    beta_start: float = 1e-10
    beta_end: float = 1e-9
    cosine_s: float = 0.008
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 0
    epochs: int = 200
    grad_clip: float = None
    ema_decay: float = 0.999



# ----------------- argparse -----------------
def get_args():
    parser = argparse.ArgumentParser(description="Run inference with config options.")

    # data / model paths
    parser.add_argument("--data_mode", type=str, default="rq3",
                        help="Mode for SliceDataLoader")
    parser.add_argument("--data_label", type=str, default="cluster",
                        help="Label type for SliceDataLoader")
    parser.add_argument("--location_model_checkpoint", type=str,
                        default="model_checkpoints/best_model_intra2hole.pt",
                        help="Path to trained location checkpoint")
    parser.add_argument("--cluster_model_checkpoint", type=str,  ### CHANGE
                        default="experimentation/best_model_rq3.pt",
                        help="Path to trained CelltypeModel checkpoint")    
    parser.add_argument("--expression_model_checkpoint", type=str,  ### CHANGE
                            default="/compute/oven-0-13/skrieger/Zhuang-2/epoch120_model.pt",
                            help="Path to trained expression checkpoint")

    parser.add_argument("--location_inference_type", type=str, default="skip",
                        help="Type of location inference")
    parser.add_argument("--kde_bandwidth", type=float, default=0.01,
                        help="Bandwidth for KDE")

    parser.add_argument("--cluster_inference_type", type=str,
                        default="skip",
                        help="How to infer subclass")
    parser.add_argument("--expression_inference_type", type=str, default="model",
                        help="How to infer gene expression")


    # training hyperparams
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs for CelltypeModel")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for CelltypeModel")
    parser.add_argument("--guidance_signal", type=float, default=0.01,
                        help="Guidance signal for classifier-based guidance")

    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for CelltypeModel")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cpu/cuda)")

    parser.add_argument(
        "--metrics",
        type=lambda s: s.split(","),
        help="Comma-separated list of metrics to compute (soft_accuracy,soft_correlation,neighborhood_enrichment,soft_precision)",
        default=["soft_f1","soft_correlation"]#,"soft_accuracy", "soft_correlation", "neighborhood_enrichment", "soft_precision"],
    )
    parser.add_argument("--metric_sampling", type=int, default=1, 
                        help="Percentage of samples to use for metric computation")
    parser.add_argument("--out_csv", type=str,
                        default="results/debugging_rq3_spencer.csv",
                        help="Output CSV file path")


    return parser.parse_args()

# ----------------- util -----------------
def write_row(row, path):
    header = not os.path.isfile(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)

def already_done(cfg, path):
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path, usecols=cfg.keys())
    return any((df == pd.Series(cfg)).all(axis=1))

# ----------------- main -----------------

def main():
    args = get_args()
    cfg = args.__dict__

    # data_for_model_init = SliceDataLoader(mode="rq1", label=args.data_label)
    # data_for_model_init.prepare()

    slice_data_loader = SliceDataLoader(mode=args.data_mode, label=args.data_label)
    slice_data_loader.prepare()

    
    # data_for_model_init=slice_data_loader  # use same references for both models

    temp_test_slices = slice_data_loader.test_slices.copy()
    temp_ref_slices = slice_data_loader.reference_slices.copy()
    try:
        temp_hole_centers = slice_data_loader.hole_centers.copy()
    except:
        pass


    for i,slice in enumerate(temp_test_slices):
        # slice = slice[np.random.choice(slice.n_obs, 1000, replace=False)]            
        #create a new folder in artificats with the timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = f"artifacts/{timestamp}"
        cfg["artifact_dir"] = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)

        slice_data_loader.test_slices = [slice]
        try:
            slice_data_loader.hole_centers = [temp_hole_centers[i]]
        except:
            pass

        slice_data_loader.reference_slices = temp_ref_slices[2*i:2*i+2]  # adjust based on which references you want to use
        cfg["slice_index"] = i

        

        traincfg = TrainConfig(
            degree=7,
            activation="silu",
            batchnorm=False,
            dropout=0.0,
            feature_type="poly",
            n_timesteps=70,
            batch_size=4096*50,
            lr=2e-4,
            epochs=1000000,
        )

        # wandb.init(entity="tissue-generator", project="ddpm-training")

        

        trainer = DDPMTrainer(None, traincfg)

        ckpt = torch.load("/compute/oven-0-13/aj_checkpoints/full_ddpm_checkpoint_4_3600.pt", map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.ema.shadow = ckpt["ema"]

        closest_ref_slice = np.argmin([np.square(ref_slice.obsm["aligned_spatial"].mean(0)[-1] - slice_data_loader.test_slices[0].obsm["aligned_spatial"].mean(0)[-1]) for ref_slice in slice_data_loader.reference_slices])    
        best_ref_slice = slice_data_loader.reference_slices[closest_ref_slice].copy()
        best_ref_slice.obsm["aligned_spatial"][:,-1] = slice_data_loader.test_slices[0].obsm["aligned_spatial"][:,-1].mean(0)
        location_model = BiologicalModel2([best_ref_slice],bandwidth=args.kde_bandwidth)
        location_model.fit()

        # celltype_model = CelltypeModel(
        #     data_for_model_init.train_slices,
        #     data_for_model_init.gene_exp_model.num_tokens,
        #     val_slice=data_for_model_init.val_slices[0],
        #     epochs=args.epochs,
        #     learning_rate=args.learning_rate,
        #     batch_size=args.batch_size,
        #     device=args.device
        # )

        celltype_model = SkeletonCelltypeModel(5274, num_features=3)

        celltype_model.load_model(args.cluster_model_checkpoint)

        if already_done(cfg, args.out_csv):
            print("skip", cfg)
            return

        inf = Inferernce((trainer, location_model), celltype_model, slice_data_loader, copy.deepcopy(cfg))
        pred = inf.run_inference(slice_data_loader.test_slices)
        res = Evaluator(cfg).evaluate(pred, slice_data_loader.test_slices[0], sample=args.metric_sampling)
        res = {k: float(v) for k, v in res.items()}

        row = {**cfg, **res}
        write_row(row, args.out_csv)
        print("wrote", cfg)
        #save config to artifact dir as json

        # save config + results into artifact dir
        cfg_path = os.path.join(artifact_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        res_path = os.path.join(artifact_dir, "results.json")
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

        # optionally also save predictions
        pred_path = os.path.join(artifact_dir, "pred.pkl")
        with open(pred_path, "wb") as f:
            pkl.dump(pred, f)        


        # delete all local variables and collect garbage
        del pred, trainer, location_model, celltype_model, inf
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    main()
