import argparse
from email import parser
import sys
sys.path.append("generative_transformer")

from celltype_model import CelltypeModel
from data_loader import SliceDataLoader
from biological_model import BiologicalModel2
import torch
import numpy as np
from inference import Inferernce
from evaluator import Evaluator
import copy, os, pandas as pd

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ----------------- argparse -----------------
def get_args():
    parser = argparse.ArgumentParser(description="Run inference with config options.")

    # data / model paths
    parser.add_argument("--data_mode", type=str, default="intra2_hole",
                        help="Mode for SliceDataLoader")
    parser.add_argument("--data_label", type=str, default="subclass",
                        help="Label type for SliceDataLoader")
    parser.add_argument("--location_model_checkpoint", type=str,
                        default="model_checkpoints/best_model_intra2hole.pt",
                        help="Path to trained location checkpoint")
    parser.add_argument("--cluster_model_checkpoint", type=str,  ### CHANGE
                        default="model_checkpoints/best_model_intra2hole.pt",
                        help="Path to trained CelltypeModel checkpoint")    
    parser.add_argument("--expression_model_checkpoint", type=str,  ### CHANGE
                            default="model_checkpoints/best_model_intra2hole.pt",
                            help="Path to trained expression checkpoint")
    parser.add_argument("--out_csv", type=str,
                        default="inference_results_intra2_trial.csv",
                        help="Output CSV file path")

    parser.add_argument("--location_inference_type", type=str, default="model",
                        help="Type of location inference")
    parser.add_argument("--cluster_inference_type", type=str,
                        default="majority_baseline",
                        help="How to infer subclass")
    parser.add_argument("--expression_inference_type", type=str, default="model",
                        help="How to infer gene expression")

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs for CelltypeModel")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for CelltypeModel")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for CelltypeModel")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cpu/cuda)")

    parser.add_argument(
        "--metrics",
        type=lambda s: s.split(","),
        help="Comma-separated list of metrics to compute (soft_accuracy,soft_correlation,neighborhood_enrichment,soft_precision)",
        default=["weighted_l1_distance","soft_accuracy", "soft_correlation", "neighborhood_enrichment", "soft_precision"],
    )


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

    slice_data_loader = SliceDataLoader(mode=args.data_mode, label=args.data_label)
    slice_data_loader.prepare()

    location_model = BiologicalModel2(slice_data_loader.train_slices)
    location_model.fit()

    celltype_model = CelltypeModel(
        slice_data_loader.train_slices,
        slice_data_loader.gene_exp_model.num_tokens,
        val_slice=slice_data_loader.val_slices[0],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device
    )

    # celltype_model.load_model(args.cluster_model_checkpoint)

    if already_done(cfg, args.out_csv):
        print("skip", cfg)
        return

    inf = Inferernce(location_model, celltype_model, slice_data_loader, copy.deepcopy(cfg))
    pred = inf.run_inference(slice_data_loader.test_slices)
    res = Evaluator(cfg).evaluate(pred, slice_data_loader.test_slices[0], sample=1)

    row = {**cfg, **res}
    write_row(row, args.out_csv)
    print("wrote", cfg)


if __name__ == "__main__":
    main()
