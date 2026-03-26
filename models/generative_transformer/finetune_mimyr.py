#!/usr/bin/env python3
"""
finetune.py

Fine-tune MimyrModel on conditional generation with coordinate tokens,
including a held-out validation split.
"""
import os
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import scanpy as sc
from tqdm import tqdm
import wandb
import torch.distributed as dist
import scipy.sparse as sp
import pandas as pd
from anndata import AnnData


from .model.model import MimyrConfig, MimyrModel
from .utils.hf_tokenizer import MimyrTokenizer
from .data_util import get_generation_dataloader, harmonize_dataset, summarize_sample_ddp
from .Mimyr import tokens_and_vals_to_expression_row

from scipy.stats import pearsonr
import numpy as np
from collections import defaultdict


def load_sampling_metadata_csv(
    adata: AnnData, path: str, overwrite: bool = True
) -> None:
    """
    Load the CSV saved above and merge into adata.obs by obs_names.
    """
    df = pd.read_csv(path)
    WEIGHT_COLS = ["class_weight", "spatial_bin", "cell_weight", "sampling_prob"]
    if "obs_name" not in df.columns:
        raise ValueError("CSV must contain an 'obs_name' column.")
    df = df.set_index("obs_name")

    # keep only rows present in adata
    df = df.loc[df.index.intersection(adata.obs_names)]
    if df.empty:
        print("No overlapping obs_names; nothing to merge.")
        return

    for c in df.columns:
        if c not in WEIGHT_COLS:
            continue
        if overwrite or c not in adata.obs.columns:
            adata.obs.loc[df.index, c] = df[c].values
        else:
            # fill NaNs only
            m = adata.obs.index.isin(df.index) & adata.obs[c].isna()
            adata.obs.loc[m, c] = df.loc[adata.obs.index[m], c].values
    print(f"Merged {list(df.columns)} for {df.shape[0]} cells from {path}")


def rebalance_sampling_mass(
    adata_train,
    group_col="dataset_type",
    sampling_col="sampling_prob",
    target_fracs=None,
    fill_missing_with=1.0,
    eps=1e-12,
):
    """
    Rescales sampling_prob so that total probability mass per group matches target_fracs.
    Keeps relative weights within each group unchanged.
    """
    if sampling_col not in adata_train.obs.columns:
        adata_train.obs[sampling_col] = fill_missing_with

    p = adata_train.obs[sampling_col].astype(float).to_numpy()
    p = np.nan_to_num(p, nan=fill_missing_with, posinf=fill_missing_with, neginf=0.0)
    adata_train.obs[sampling_col] = p

    if target_fracs is None:
        groups = adata_train.obs[group_col].astype("category")
        cats = [c for c in groups.cat.categories if (groups == c).any()]
        target = {c: 1.0 / max(len(cats), 1) for c in cats}
    else:
        target = target_fracs

    df = adata_train.obs[[group_col, sampling_col]].copy()
    cur = df.groupby(group_col)[sampling_col].sum().to_dict()

    scales = {}
    for g, t in target.items():
        m = max(cur.get(g, 0.0), eps)
        scales[g] = t / m

    gvals = adata_train.obs[group_col].to_numpy()
    scale_vec = np.vectorize(lambda g: scales.get(g, 1.0))(gvals)
    adata_train.obs[sampling_col] = adata_train.obs[sampling_col].to_numpy() * scale_vec
    print("rebalanced sampling probs")
    return scales


def evaluate_expression_correlation(
    adata,
    batch,
    logits_labels,
    logits_exp_real,
    tokenizer,
    tokens_and_vals_to_expression_row_fn,
    var_names,
    labels,
    mask_token_id=-100,
    threshold=0.0,
):
    """
    Compute per-cell Pearson correlation between predicted and true expression.

    Parameters
    ----------
    adata : AnnData
        The original data object to get ground truth from.
    batch : dict
        The current batch from the dataloader.
    logits_labels : Tensor
        Output of the model (B, T, vocab_size) — token logits.
    logits_exp_real : Tensor
        Output of the model (B, T, 1) — real-valued expression predictions.
    tokenizer : MimyrTokenizer
        Tokenizer used to map tokens to strings.
    tokens_and_vals_to_expression_row_fn : function
        The function that builds a gene expression vector from tokens and values.
    var_names : List[str]
        List of all gene names in order.
    mask_token_id : int
        The ID of the mask/pad token to ignore in output.

    Returns
    -------
    pearson_rs : List[float]
        Pearson correlation per cell in the batch.
    """
    B, T, V = logits_labels.shape
    logits_cls = logits_labels.argmax(dim=-1).cpu().numpy()  # (B, T)
    expr_vals = logits_exp_real.squeeze(-1).cpu().numpy()  # (B, T)
    if hasattr(labels, "detach"):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    idxs = batch["idx"] if "idx" in batch else range(B)  # indices into adata

    predicted_expr_rows = []
    ground_truth_rows = []

    for b in range(B):
        gene_token_ids = logits_cls[b]
        expr_values = expr_vals[b]
        labels_b = labels_np[b]  # (T,)

        # Filter out special tokens (mask, EOS, etc.)
        valid_mask = labels_b != mask_token_id
        gene_token_ids = gene_token_ids[valid_mask]
        expr_values = expr_values[valid_mask]

        # Convert token IDs to strings
        gene_tokens = tokenizer.convert_ids_to_tokens(gene_token_ids.tolist())
        # if b == 0:
        #     print('labels')
        #     print(labels_b)
        #     print(valid_mask)
        #     print('gene token ids')
        #     print(gene_token_ids)
        #     print('gene tokens')
        #     print(gene_tokens)
        #     print('expression values:')
        #     print(expr_values)

        # Construct predicted expression row
        pred_expr = tokens_and_vals_to_expression_row_fn(
            var_names=var_names,
            gene_tokens=gene_tokens,
            gene_tokens_int=gene_token_ids.tolist(),
            new_vals=expr_values.tolist(),
            return_series=False,
        )
        # if b == 0:
        #     print('pred_expr')
        #     print(pred_expr)
        #     for v, p in zip(var_names,pred_expr):
        #         if p > 0:
        #             print(f'{v}: {p}')
        predicted_expr_rows.append(pred_expr)

        # Get true expression from AnnData (dense or sparse)
        gt_expr = (
            adata.X[idxs[b]].toarray().flatten()
            if hasattr(adata.X, "toarray")
            else adata.X[idxs[b]]
        )
        ground_truth_rows.append(gt_expr)

    # Compute per-cell Pearson r
    pearson_rs = []
    for pred, gt in zip(predicted_expr_rows, ground_truth_rows):
        if np.std(pred) > 0 and np.std(gt) > 0:
            r, _ = pearsonr(pred, gt)
        else:
            r = 0.0
        pearson_rs.append(r)

    return pearson_rs


def evaluate_expression_metrics(
    adata,
    batch,
    logits_labels,  # (B, T, V) torch.Tensor
    logits_exp_real,  # (B, T, 1) torch.Tensor
    tokenizer,
    tokens_and_vals_to_expression_row_fn,
    var_names,
    labels,  # (B, T) torch.Tensor or np.ndarray; ignore positions == mask_token_id
    mask_token_id=-100,
    threshold=0.0,  # expression > threshold => positive
    verbose=False,
):
    """
    Returns per-cell metrics: Pearson r, F1, precision, recall.
    """
    import numpy as np
    from scipy.stats import pearsonr

    gene_sums = np.asarray(adata.X.sum(axis=0)).ravel()

    # Align sums to provided var_names (in case orders differ)
    # Fast path if identical:
    if list(getattr(adata, "var_names", [])) == list(var_names):
        valid_gene_mask = gene_sums > 0
    else:
        # Map adata.var_names -> index
        name_to_idx = {g: i for i, g in enumerate(list(adata.var_names))}
        valid_gene_mask = np.array(
            [
                (gene_sums[name_to_idx[g]] > 0) if g in name_to_idx else False
                for g in var_names
            ],
            dtype=bool,
        )

    B, T, V = logits_labels.shape
    logits_cls = logits_labels.argmax(dim=-1).detach().cpu().numpy()  # (B, T)
    expr_vals = logits_exp_real.squeeze(-1).detach().cpu().numpy()  # (B, T)

    if hasattr(labels, "detach"):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    idxs = batch.get("idx", range(B))

    pearson_rs, f1s, precisions, recalls = [], [], [], []
    for b in range(B):
        ids = logits_cls[b]
        vals = expr_vals[b]
        lab = labels_np[b]

        # Keep only positions we trained on (mask out prompt/pad/etc.)
        keep = lab != mask_token_id
        if not np.any(keep):
            # No usable tokens -> zero-vector prediction
            pred_expr = np.zeros(len(var_names), dtype=float)
        else:
            ids = ids[keep]
            vals = vals[keep]
            tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            pred_expr = tokens_and_vals_to_expression_row_fn(
                var_names=var_names,
                gene_tokens=tokens,
                gene_tokens_int=ids.tolist(),
                new_vals=vals.tolist(),
                return_series=False,
                max_violations=10,
            )

        # Ground truth
        gt = adata.X[idxs[b]]
        gt = gt.toarray().ravel() if hasattr(gt, "toarray") else np.asarray(gt).ravel()

        pred_expr = pred_expr[valid_gene_mask]
        gt = gt[valid_gene_mask]
        valid_genes = np.array(var_names)[valid_gene_mask]

        # Pearson r
        if np.std(pred_expr) > 0 and np.std(gt) > 0:
            r, _ = pearsonr(pred_expr, gt)
        else:
            r = 0.0
        pearson_rs.append(float(r))

        # F1 / Precision / Recall (binary by threshold)
        pred_pos = pred_expr > threshold
        gt_pos = gt > threshold

        tp = int(np.count_nonzero(pred_pos & gt_pos))
        pp = int(np.count_nonzero(pred_pos))
        gp = int(np.count_nonzero(gt_pos))

        prec = tp / pp if pp > 0 else 0.0
        rec = tp / gp if gp > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1))
        if b == 0 and verbose == True:
            print("\n================ DEBUG: Cell 0 ================")
            print("labels")
            print(lab)
            print("gene token ids")
            print(ids)
            print("gene tokens")
            print(tokens)
            print("expression values:")
            print(vals)
            print("pred_expr")
            print(pred_expr)
            # for v, p in zip(var_names[valid_gene_mask],pred_expr):
            #     if p > 0:
            #         print(f'{v}: {p}')

            gt_expressed_idx = np.where(gt > 0)[0]
            print(
                f"Ground-truth expressed genes: {len(gt_expressed_idx)} / {len(valid_genes)}"
            )

            for i in gt_expressed_idx:
                gene = valid_genes[i]
                gt_val = gt[i]
                pred_val = pred_expr[i]

                present_in_pred = gene in tokens
                pred_idx = tokens.index(gene) if gene in tokens else -1
                pred_val_raw = vals[pred_idx] if pred_idx != -1 else 0.0
                flag = "✅" if present_in_pred else "❌"
                print(
                    f"{flag} {gene:15s} | GT={gt_val:.3f} | Pred={pred_val:.3f} | Pred2={pred_val_raw:.3f}"
                )

            # Genes predicted >0 but GT==0
            false_pos_idx = np.where((gt == 0) & (pred_expr > 0))[0]
            false_pos_genes = valid_genes[false_pos_idx]
            if len(false_pos_genes) > 0:
                print("\nPredicted >0 but GT=0 (false positives):")
                print(", ".join(false_pos_genes.tolist()))
            else:
                print("\nNo false positives detected.")
            print("================================================\n")

    return pearson_rs, f1s, precisions, recalls


def get_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune MimyrModel on conditional generation with coords + validation"
    )
    parser.add_argument(
        "--ckp-path", type=str, default=None, help="Path to pretrained checkpoint (.pt)"
    )
    parser.add_argument(
        "--meta-info",
        type=str,
        required=True,
        help="Path to meta_info.pt from pretraining",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        required=True,
        help="Data mode to pass to SliceDataLoader (e.g. 'rq1', 'rq2', 'rq3', 'rq4', 'rq5')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing the .h5ad data files",
    )
    parser.add_argument(
        "--data-label",
        type=str,
        default="cluster",
        help="Cell-type label column to use in SliceDataLoader (default: 'cluster')",
    )
    parser.add_argument(
        "--adata2",
        type=str,
        default=None,
        help="Path to second input AnnData .h5ad file",
    )
    parser.add_argument(
        "--kv-cache", action="store_true", help="Whether to use kv-cached model variant"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save finetuned model and tokenizer",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of fine-tuning epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Max sequence length for prompts + genes",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Disable DataLoader shuffling"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=10,
        help="Number of epochs between model checkpoints",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=1.0,
        help="Weight on expression MSE loss term",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cpu or cuda)",
    )
    parser.add_argument(
        "--from-finetuned",
        action="store_true",
        help="Indicate checkpoint already includes finetuned vocab size",
    )
    parser.add_argument(
        "--overwrite-vocab-size",
        type=int,
        default=None,
        help="If set, overwrite the model and config vocab_size to this value before loading state_dict",
    )
    parser.add_argument(
        "--new-expression-size",
        type=int,
        default=None,
        help="If set, overwrite the model and config n_expression_level to this value",
    )
    parser.add_argument(
        "--xyz-noise",
        action="store_true",
        help="Whether to add noise to x,y,z coordinates during training",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Whether to use small dummy dataset"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default=None,
        help="Model size to initialize if no checkpoint is provided",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="/work/magroup/skrieger/tissue_generator/spencer_gentran/generative_transformer/metadata/",
        help="Directory containing metadata files (edges, meta_info) for harmonization",
    )
    parser.add_argument(
        "--disable-sampling-probs",
        action="store_true",
        help="Ignore adata.obs[sampling_col] even if present (uniform sampling).",
    )
    parser.add_argument(
        "--rebalance-only",
        action="store_true",
        help="Ignore CSV weights; set uniform per-cell weights then rebalance dataset mass across groups (st vs scrna).",
    )
    parser.add_argument(
        "--sampling-col",
        type=str,
        default="sampling_prob",
        help="obs column name holding per-cell sampling probabilities",
    )
    parser.add_argument(
        "--epoch-samples",
        type=int,
        default=-1,
        help="Rows to draw per epoch (set -1 to use len(dataset))",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-per-steps",
        type=int,
        default=100,
        help="Logging to WANDB only after this many steps",
    )
    parser.add_argument(
        "--bin-edges-file",
        type=str,
        default=None,
        help="bin_edges file for gene expression binning",
    )

    return parser


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if dist.is_available() and dist.is_initialized():
        print("using a distributed multi-gpu run\n\n\n")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        device = torch.device(args.device)

    # 1) Load pretrained checkpoint
    if args.ckp_path is not None:
        ckp = torch.load(args.ckp_path, map_location="cpu")
    else:
        # Choose default config based on --model-size
        if args.model_size is None:
            raise ValueError(
                "You must specify --model-size if --ckp_path is not provided"
            )

        if args.model_size == "small":
            model_args = {
                "n_embd": 64,
                "n_layer": 2,
                "n_head": 4,
                "vocab_size": 1011,
                "expression_level": 100,
                "dropout": 0.1,
                "ele": 1,
            }
        elif args.model_size == "medium":
            model_args = {
                "n_embd": 384,
                "n_layer": 12,
                "n_head": 12,
                "vocab_size": 1011,
                "expression_level": 100,
                "dropout": 0.1,
                "ele": 1,
            }
        elif args.model_size == "large":
            model_args = {
                "n_embd": 1120,
                "n_layer": 24,
                "n_head": 16,
                "vocab_size": 1011,
                "expression_level": 100,
                "dropout": 0.1,
                "ele": 1,
            }
        else:
            raise ValueError("Invalid --model-size. Choose from: small, medium, large")

        ckp = {"model_args": model_args}
    if args.overwrite_vocab_size is not None:
        ckp["model_args"]["vocab_size"] = args.overwrite_vocab_size
        print(f"Overwriting config.vocab_size to {args.overwrite_vocab_size}")
    if args.new_expression_size is not None and args.from_finetuned:
        ckp["model_args"]["expression_level"] = args.new_expression_size
    gptconf = MimyrConfig(**ckp["model_args"])
    print(gptconf)
    ModelClass = MimyrModel
    model = ModelClass(gptconf)
    # device = torch.device(args.device)

    if args.from_finetuned:
        model.load_state_dict(ckp["model"], strict=False)
    model.eval()
    model.hidden_dim = ckp["model_args"]["n_embd"]

    meta_info = torch.load(args.meta_info)
    print("loaded meta_info")

    # 3) Initialize tokenizer and resize model embeddings/output
    tokenizer = MimyrTokenizer(meta_info["token_set"])
    if not args.from_finetuned:
        sep = meta_info.get("sep_token", "<SPToken1>")
        tokenizer.add_special_tokens({"sep_token": sep})
        # resize_token_embeddings comes from PreTrainedModel
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        ckp["model_args"]["vocab_size"] = len(tokenizer)
        if args.new_expression_size:
            model.resize_expression_embeddings(args.new_expression_size)
            model.config.expression_level = args.new_expression_size
            ckp["model_args"]["expression_level"] = args.new_expression_size

    print("initialized model", flush=True)

    # 4) Load train/val AnnData from data_loader
    from data_loader import SliceDataLoader

    cfg = {
        "data_dir": args.data_dir,
        "meta_info": os.path.basename(args.meta_info),
    }
    slice_loader = SliceDataLoader(
        mode=args.data_mode,
        label=args.data_label,
        cfg=cfg,
        metadata_dir=args.metadata_dir,
    )
    slice_loader.prepare(
        adata2_path=args.adata2,
        gene_set=meta_info.get("gene_set"),
        seed=args.seed,
        dummy=args.dummy,
        sampling_col=args.sampling_col,
        output_dir=args.output_dir,
        rank=rank,
    )

    adata_train = slice_loader.adata_train
    adata_val = slice_loader.adata_val

    args.use_sampling_probs = not args.disable_sampling_probs
    if args.rebalance_only:
        print("Mode: --rebalance-only. Using uniform per-cell weights then rebalancing dataset mass.")
        adata_train.obs[args.sampling_col] = 1.0
        args.use_sampling_probs = True
        if "dataset_type" in adata_train.obs.columns:
            rebalance_sampling_mass(
                adata_train,
                group_col="dataset_type",
                sampling_col=args.sampling_col,
                target_fracs={"st": 0.7, "scrna": 0.3},
            )
    elif args.use_sampling_probs:
        load_sampling_metadata_csv(adata_train, "train_adata_sampling.csv.gz", overwrite=True)

    use_probs_val = False  # usually evaluate uniformly

    # Optional: if you want to limit rows per epoch, pass args.epoch_samples if you have it
    epoch_samples = getattr(args, "epoch_samples", None)
    base_seed = getattr(args, "seed", 0)

    if args.bin_edges_file is not None:
        bin_edges = torch.load(args.bin_edges_file)
    else:
        bin_edges = None

    train_loader, bin_edges = get_generation_dataloader(
        adata=adata_train,
        meta_info=meta_info,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        n_express_level=model.config.expression_level,
        include_0s=False,
        add_xyz_noise=args.xyz_noise,
        # exclude_columns = ['supertype','cluster'],
        use_sampling_probs=args.use_sampling_probs,
        sampling_col=args.sampling_col,
        epoch_samples=epoch_samples,
        seed=base_seed,
        bin_edges=bin_edges,
    )
    if not args.bin_edges_file:
        torch.save(bin_edges, f"{args.output_dir}bin_edges.pt")

    if adata_val is not None:
        val_loader, _ = get_generation_dataloader(
            adata=adata_val,
            meta_info=meta_info,
            batch_size=args.batch_size,
            max_len=args.max_len,
            shuffle=False,
            num_workers=args.num_workers,
            n_express_level=model.config.expression_level,
            include_0s=False,
            # keep uniform sampling for validation
            use_sampling_probs=use_probs_val,
            sampling_col=args.sampling_col,
            bin_edges=bin_edges,
        )
    else:
        val_loader = None
    print("loaded anndata")

    # 5) Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.to(device)
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    # 6) Initialize W&B
    if rank == 0:
        wandb.init(
            project="Mimyr-finetune",
            name=f"{os.path.basename(args.output_dir)}_bs{args.batch_size}",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lambda_val": args.lambda_val,
            },
            dir=args.output_dir,
        )

    if args.from_finetuned:
        # e.g. ckp-path ends with ".../epoch3_model.pt"
        import re

        m = re.search(r"epoch(\d+)", os.path.basename(args.ckp_path))
        start_epoch = int(m.group(1)) + 1 if m else 1
    else:
        start_epoch = 1

    # 7) Training + validation loop
    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Grab the *local* indices from the sampler, then aggregate
        local_idx = None
        if hasattr(train_loader.sampler, "get_last_indices"):
            # Need to trigger one iterator to make __iter__ run at least once
            _ = iter(train_loader)
            local_idx = train_loader.sampler.get_last_indices()
            summarize_sample_ddp(
                adata_train,
                local_idx,
                cell_key="cluster",
                group_key="dataset_type",
                top_k=20,
            )

        model.train()
        total_loss, total_cls, total_exp_bin, total_exp_real = 0.0, 0.0, 0.0, 0.0
        all_train_pearson_rs = []
        all_train_f1, all_train_prec, all_train_rec = [], [], []
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [train]")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            x_expr = batch["input_vals"].to(device)
            expr_target = batch["target_vals"].to(device)

            if labels is not None:
                labels = labels.to(device)

            optimizer.zero_grad()
            (
                logits_cls,
                logits_exp_bins,
                logits_exp_real,
                loss,
                loss_cls,
                loss_exp_bin,
                loss_exp_real,
            ) = model(
                idx=input_ids,
                x_expr=x_expr,
                targets=labels,
                y_expr=expr_target,
                lambda_val=args.lambda_val,
                return_hidden=False,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() if loss is not None else 0.0
            total_cls += loss_cls.item() if loss_cls is not None else 0.0
            total_exp_bin += loss_exp_bin.item() if loss_exp_bin is not None else 0.0
            total_exp_real += loss_exp_real.item() if loss_exp_real is not None else 0.0

            if rank == 0:
                if step % args.log_per_steps == 0:
                    with torch.no_grad():
                        pearson_rs_train, f1s, precs, recs = (
                            evaluate_expression_metrics(
                                adata=adata_train,
                                batch=batch,
                                logits_labels=logits_cls.detach().cpu(),
                                logits_exp_real=logits_exp_real.detach().cpu(),
                                tokenizer=tokenizer,
                                tokens_and_vals_to_expression_row_fn=tokens_and_vals_to_expression_row,
                                var_names=adata_train.var_names.tolist(),
                                labels=labels,  # tensor on device is fine; fn handles .detach().cpu()
                                mask_token_id=-100,
                            )
                        )
                        mean_train_pearson_r = (
                            float(np.mean(pearson_rs_train))
                            if len(pearson_rs_train)
                            else 0.0
                        )
                        all_train_pearson_rs.extend(pearson_rs_train)
                        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
                        mean_prc = float(np.mean(precs)) if precs else 0.0
                        mean_rec = float(np.mean(recs)) if recs else 0.0
                        all_train_f1.extend(f1s)
                        all_train_prec.extend(precs)
                        all_train_rec.extend(recs)

                    wandb.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/batch_loss_cls": loss_cls.item(),
                            "train/batch_loss_exp_bin": loss_exp_bin.item(),
                            "train/batch_loss_exp_real": loss_exp_real.item(),
                            "train/pearson_r": mean_train_pearson_r,
                            "train/f1": mean_f1,
                            "train/precision": mean_prc,
                            "train/recall": mean_rec,
                            "train/step": (epoch - 1) * len(train_loader) + step,
                        }
                    )

        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls / len(train_loader)
        avg_exp_bin = total_exp_bin / len(train_loader)
        avg_exp_real = total_exp_real / len(train_loader)
        print(
            f"Epoch {epoch} — train total {avg_loss:.4f}, cls {avg_cls:.4f}, exp_bin {avg_exp_bin:.4f}, exp_real {avg_exp_real:.4f}"
        )
        if rank == 0:
            mean_train_pearson_epoch = (
                float(np.mean(all_train_pearson_rs))
                if len(all_train_pearson_rs)
                else 0.0
            )
            mean_train_f1_epoch = float(np.mean(all_train_f1)) if all_train_f1 else 0.0
            mean_train_prc_epoch = (
                float(np.mean(all_train_prec)) if all_train_prec else 0.0
            )
            mean_train_rec_epoch = (
                float(np.mean(all_train_rec)) if all_train_rec else 0.0
            )
            wandb.log(
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_loss_cls": avg_cls,
                    "train/epoch_loss_exp_bin": avg_exp_bin,
                    "train/epoch_loss_exp_real": avg_exp_real,
                    "train/pearson_r_epoch": mean_train_pearson_epoch,
                    "train/f1_epoch": mean_train_f1_epoch,
                    "train/precision_epoch": mean_train_prc_epoch,
                    "train/recall_epoch": mean_train_rec_epoch,
                    "epoch": epoch,
                }
            )

        if val_loader is not None:
            torch.cuda.empty_cache()
            model.eval()
            v_loss, v_cls, v_exp_bin, v_exp_real, count = 0.0, 0.0, 0.0, 0.0, 0
            all_pearson_rs = []
            all_val_f1, all_val_prec, all_val_rec = [], [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch.get("labels")
                    x_expr = batch["input_vals"].to(device)
                    expr_target = batch["target_vals"].to(device)

                    if labels is not None:
                        labels = labels.to(device)

                    logits_labels, _, logits_exp_real, l, lc, leb, ler = model(
                        idx=input_ids,
                        x_expr=x_expr,
                        targets=labels,
                        y_expr=expr_target,
                        lambda_val=args.lambda_val,
                        return_hidden=False,
                    )
                    v_loss += l.item()
                    v_cls += lc.item()
                    v_exp_bin += leb.item()
                    v_exp_real += ler.item()
                    count += 1
                    pearson_rs, f1s, precs, recs = evaluate_expression_metrics(
                        adata=adata_val,
                        batch=batch,
                        logits_labels=logits_labels.cpu(),
                        logits_exp_real=logits_exp_real.cpu(),
                        tokenizer=tokenizer,
                        tokens_and_vals_to_expression_row_fn=tokens_and_vals_to_expression_row,
                        var_names=adata_val.var_names.tolist(),
                        labels=labels,
                        mask_token_id=-100,
                    )
                    all_pearson_rs.extend(pearson_rs)
                    all_val_f1.extend(f1s)
                    all_val_prec.extend(precs)
                    all_val_rec.extend(recs)

            avg_v_loss = v_loss / count
            avg_v_cls = v_cls / count
            avg_v_exp_bin = v_exp_bin / count
            avg_v_exp_real = v_exp_real / count
            print(
                f"Epoch {epoch} — valid total {avg_v_loss:.4f}, cls {avg_v_cls:.4f}, exp {avg_v_exp_bin:.4f}"
            )
            mean_pearson_r = np.mean(all_pearson_rs)
            print(f"Validation mean Pearson r: {mean_pearson_r:.4f}")
            mean_val_f1 = float(np.mean(all_val_f1)) if all_val_f1 else 0.0
            mean_val_prc = float(np.mean(all_val_prec)) if all_val_prec else 0.0
            mean_val_rec = float(np.mean(all_val_rec)) if all_val_rec else 0.0
            print(
                f"Validation mean F1: {mean_val_f1:.4f} (P={mean_val_prc:.4f}, R={mean_val_rec:.4f})"
            )
            if rank == 0:
                wandb.log(
                    {
                        "valid/epoch_loss": avg_v_loss,
                        "valid/epoch_loss_cls": avg_v_cls,
                        "valid/epoch_loss_exp_bin": avg_v_exp_bin,
                        "valid/epoch_loss_exp_real": avg_v_exp_real,
                        "valid/pearson_r": mean_pearson_r,
                        "valid/f1": mean_val_f1,
                        "valid/precision": mean_val_prc,
                        "valid/recall": mean_val_rec,
                        "epoch": epoch,
                    }
                )

        # Save epoch checkpoint
        if epoch % args.save_frequency == 0 and rank == 0:
            ckpt_file = os.path.join(args.output_dir, f"epoch{epoch}_model.pt")
            if dist.is_available() and dist.is_initialized():
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "model_args": ckp["model_args"],
                    },
                    ckpt_file,
                )
            else:
                torch.save(
                    {"model": model.state_dict(), "model_args": ckp["model_args"]},
                    ckpt_file,
                )

    # 8) Save final artifacts
    # model.save_pretrained(args.output_dir)
    MimyrConfig(**ckp["model_args"]).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Finetuned artifacts written to {args.output_dir}")
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


def main():
    train(get_parser().parse_args())


if __name__ == "__main__":
    main()