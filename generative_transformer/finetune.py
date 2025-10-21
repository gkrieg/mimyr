#!/usr/bin/env python3
"""
finetune.py

Fine-tune scMulanModel on conditional generation with coordinate tokens,
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

root_path = os.path.abspath('/work/magroup/skrieger/scMulan/Tutorials/scMulan')
sys.path.append(os.path.abspath(root_path))

from model.model import MulanConfig, scMulanModel
from utils.hf_tokenizer import scMulanTokenizer
from data_util import get_generation_dataloader, harmonize_dataset, summarize_sample_ddp
from scMulan import tokens_and_vals_to_expression_row

from scipy.stats import pearsonr
import numpy as np
from collections import defaultdict


def load_sampling_metadata_csv(adata: AnnData, path: str, overwrite: bool = True) -> None:
    """
    Load the CSV saved above and merge into adata.obs by obs_names.
    """
    df = pd.read_csv(path)
    WEIGHT_COLS = ['class_weight', 'spatial_bin', 'cell_weight', 'sampling_prob']
    if 'obs_name' not in df.columns:
        raise ValueError("CSV must contain an 'obs_name' column.")
    df = df.set_index('obs_name')

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

def evaluate_expression_correlation(
    adata,
    batch,
    logits_labels,
    logits_exp_real,
    tokenizer,
    tokens_and_vals_to_expression_row_fn,
    var_names,
    mask_token_id=-100
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
    tokenizer : scMulanTokenizer
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
    logits_cls = logits_labels.argmax(dim=-1).cpu().numpy()           # (B, T)
    expr_vals = logits_exp_real.squeeze(-1).cpu().numpy()             # (B, T)
    idxs = batch['idx'] if 'idx' in batch else range(B)               # indices into adata

    predicted_expr_rows = []
    ground_truth_rows = []

    for b in range(B):
        gene_token_ids = logits_cls[b]
        expr_values = expr_vals[b]
        
        # Filter out special tokens (mask, EOS, etc.)
        valid_mask = (gene_token_ids != mask_token_id)
        gene_token_ids = gene_token_ids[valid_mask]
        expr_values = expr_values[valid_mask]

        # Convert token IDs to strings
        gene_tokens = tokenizer.convert_ids_to_tokens(gene_token_ids.tolist())

        # Construct predicted expression row
        pred_expr = tokens_and_vals_to_expression_row_fn(
            var_names=var_names,
            gene_tokens=gene_tokens,
            gene_tokens_int=gene_token_ids.tolist(),
            new_vals=expr_values.tolist(),
            return_series=False
        )
        predicted_expr_rows.append(pred_expr)

        # Get true expression from AnnData (dense or sparse)
        gt_expr = adata.X[idxs[b]].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[idxs[b]]
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

def rebalance_sampling_mass(
    adata_train,
    group_col='dataset_type',          # expects values like 'st' and 'scrna'
    sampling_col='sampling_prob',
    target_fracs=None,                 # e.g., {'st': 0.5, 'scrna': 0.5}
    fill_missing_with=1.0,             # create uniform weights where missing
    eps=1e-12
):
    """
    Rescales sampling_prob so that total probability mass per group matches target_fracs.
    Keeps relative weights *within* each group unchanged.
    """
    import numpy as np
    import pandas as pd

    if sampling_col not in adata_train.obs.columns:
        adata_train.obs[sampling_col] = fill_missing_with

    # fill NaNs or non-finite
    p = adata_train.obs[sampling_col].astype(float).to_numpy()
    p = np.nan_to_num(p, nan=fill_missing_with, posinf=fill_missing_with, neginf=0.0)
    adata_train.obs[sampling_col] = p

    # If no targets provided, split equally across present groups
    if target_fracs is None:
        groups = adata_train.obs[group_col].astype('category')
        cats = [c for c in groups.cat.categories if (groups == c).any()]
        target = {c: 1.0 / max(len(cats), 1) for c in cats}
    else:
        target = target_fracs

    # current mass per group
    df = adata_train.obs[[group_col, sampling_col]].copy()
    cur = df.groupby(group_col)[sampling_col].sum().to_dict()

    # rescale per group: p_i <- p_i * (target_frac / current_mass)
    scales = {}
    for g, t in target.items():
        m = max(cur.get(g, 0.0), eps)
        scales[g] = t / m

    gvals = adata_train.obs[group_col].to_numpy()
    scale_vec = np.vectorize(lambda g: scales.get(g, 1.0))(gvals)
    adata_train.obs[sampling_col] = (adata_train.obs[sampling_col].to_numpy() * scale_vec)
    print('rebalanced sampling probs')

    # (optional) final global normalization is done inside the dataloader anyway
    return scales



def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune scMulanModel on conditional generation with coords + validation"
    )
    parser.add_argument("--ckp-path",    type=str, default=None,
                        help="Path to pretrained checkpoint (.pt)")
    parser.add_argument("--meta-info",   type=str, required=True,
                        help="Path to meta_info.pt from pretraining")
    parser.add_argument("--adata",       type=str, required=True,
                        help="Path to input AnnData .h5ad file")
    parser.add_argument("--adata2",       type=str, default=None,
                        help="Path to second input AnnData .h5ad file")
    parser.add_argument("--kv-cache",    action="store_true",
                        help="Whether to use kv-cached model variant")
    parser.add_argument("--output-dir",  type=str, required=True,
                        help="Directory to save finetuned model and tokenizer")
    parser.add_argument("--epochs",      type=int, default=5,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr",          type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max-len",     type=int, default=512,
                        help="Max sequence length for prompts + genes")
    parser.add_argument("--no-shuffle",  action="store_true",
                        help="Disable DataLoader shuffling")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--save-frequency", type=int, default=10,
                        help="Number of epochs between model checkpoints")
    parser.add_argument("--lambda-val",  type=float, default=1.0,
                        help="Weight on expression MSE loss term")
    parser.add_argument("--val-split",   type=float, default=0.1,
                        help="Fraction of cells to hold out for validation (0 disables)")
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for training (cpu or cuda)")
    parser.add_argument("--from-finetuned", action="store_true",
                        help="Indicate checkpoint already includes finetuned vocab size")
    parser.add_argument("--overwrite-vocab-size", type=int, default=None,
                        help="If set, overwrite the model and config vocab_size to this value before loading state_dict")
    parser.add_argument("--new-expression-size", type=int, default=None,
                        help="If set, overwrite the model and config n_expression_level to this value")
    parser.add_argument("--xyz-noise",    action="store_true",
                        help="Whether to add noise to x,y,z coordinates during training")
    parser.add_argument("--slice-nums", nargs='+',type=str, default=None,
                        help='List of slice IDs for spatial mouse brain atlases')
    parser.add_argument(
        "--val-slice-nums", nargs='+', type=int, default=None,
        help="List of slice IDs to use for validation. If provided, we split by slices: "
             "val = these slices; train = --slice-nums (if provided) or all other slices present."
    )
    parser.add_argument("--dummy",    action="store_true",
                        help="Whether to use small dummy dataset")
    parser.add_argument("--model-size", type=str, choices=["small", "medium", "large"], default=None,
                    help="Model size to initialize if no checkpoint is provided")
    parser.add_argument("--slice-prefix",      type=str,
                        default="C57BL6J-638850.",
                        help="obs prefix for the slice nums")
    parser.add_argument("--metadata-dir",      type=str,
                        default="/work/magroup/skrieger/tissue_generator/spencer_gentran/generative_transformer/metadata/",
                        help="obs prefix for the slice nums")
    parser.add_argument("--harmonize-dataset",    action="store_true",
                        help="Whether to add all gene tokens and normalize")
    parser.add_argument("--technology",      type=str,
                        default="M500",
                        help="technology of data")
    parser.add_argument("--coord-suffix",      type=str,
                        default="_ccf",
                        help="suffix in .obs for CCF coordinates")
    parser.add_argument('--disable-sampling-probs', action='store_true',
                        help='Ignore adata.obs[sampling_col] even if present (uniform sampling).')
    parser.add_argument('--sampling-col', type=str, default='sampling_prob',
                        help='obs column name holding per-cell sampling probabilities')
    parser.add_argument('--epoch-samples', type=int, default=-1,
                        help='Rows to draw per epoch (set -1 to use len(dataset))')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-per-steps', type=int, default=100,
                        help='Logging to WANDB only after this many steps')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['NCCL_P2P_DISABLE'] = '1'
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend='nccl')


    if dist.is_available() and dist.is_initialized():
        print('using a distributed multi-gpu run\n\n\n')
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        rank = 0
        device = torch.device(args.device)



    # 1) Load pretrained checkpoint
    if args.ckp_path is not None:
        ckp = torch.load(args.ckp_path, map_location='cpu')
    else:
        # Choose default config based on --model-size
        if args.model_size is None:
            raise ValueError("You must specify --model-size if --ckp_path is not provided")
    
        if args.model_size == "small":
            model_args = {
                'n_embd': 64,
                'n_layer': 2,
                'n_head': 4,
                'vocab_size': 1011,
                'expression_level': 100,
                'dropout': 0.1,
                'ele': 1
            }
        elif args.model_size == "medium":
            model_args = {
                'n_embd': 384,
                'n_layer': 12,
                'n_head': 12,
                'vocab_size': 1011,
                'expression_level': 100,
                'dropout': 0.1,
                'ele': 1
            }
        elif args.model_size == "large":
            model_args = {
                'n_embd': 1120,
                'n_layer': 24,
                'n_head': 16,
                'vocab_size': 1011,
                'expression_level': 100,
                'dropout': 0.1,
                'ele': 1
            }
        else:
            raise ValueError("Invalid --model-size. Choose from: small, medium, large")
    
        ckp = {'model_args': model_args}
    if args.overwrite_vocab_size is not None:
        ckp['model_args']['vocab_size'] = args.overwrite_vocab_size
        print(f"Overwriting config.vocab_size to {args.overwrite_vocab_size}")
    if args.new_expression_size is not None and args.from_finetuned:
        ckp['model_args']['expression_level'] = args.new_expression_size
    gptconf = MulanConfig(**ckp['model_args'])
    print(gptconf)
    ModelClass = scMulanModel
    model = ModelClass(gptconf)
    # device = torch.device(args.device)
    
    if args.from_finetuned:
        model.load_state_dict(ckp['model'], strict=False)
    model.eval()
    model.hidden_dim = ckp['model_args']['n_embd']

    meta_info = torch.load(args.meta_info)
    print('loaded meta_info')
    
    # 3) Initialize tokenizer and resize model embeddings/output
    tokenizer = scMulanTokenizer(meta_info['token_set'])
    if not args.from_finetuned:
        sep = meta_info.get('sep_token', '<SPToken1>')
        tokenizer.add_special_tokens({'sep_token': sep})
        # resize_token_embeddings comes from PreTrainedModel
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        ckp['model_args']['vocab_size'] = len(tokenizer)
        if args.new_expression_size:
            model.resize_expression_embeddings(args.new_expression_size)
            model.config.expression_level = args.new_expression_size
            ckp['model_args']['expression_level'] = args.new_expression_size
    
    
    print('initialized model', flush=True)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    # 4) Load AnnData ONCE

    def _set_uniform_sampling_prob(adata_like, col='sampling_prob'):
        """Add a uniform per-cell sampling probability (sum doesn't need to be 1)."""
        import numpy as np
        adata_like.obs[col] = np.full(adata_like.n_obs, 1.0, dtype=np.float64)  # loader will renormalize

    adata = sc.read_h5ad(args.adata)
    load_sampling_metadata_csv(adata, 'train_adata_sampling.csv.gz', overwrite=True)
    
    if args.harmonize_dataset:
        coord_files = [f'{args.metadata_dir}edges_x.pkl', f'{args.metadata_dir}edges_y.pkl', f'{args.metadata_dir}edges_z.pkl']
        adata = harmonize_dataset(adata, meta_info, coord_files, technology=args.technology, coord_suffix=args.coord_suffix)
    
    if args.slice_nums is not None and args.val_slice_nums is None:
        slice_names_train = [f'{args.slice_prefix}{s}' for s in args.slice_nums]
        adata._inplace_subset_obs(adata.obs['brain_section_label'].isin(slice_names_train))
        print(f"Subsetted adata to only contain train slices {args.slice_nums} → {adata}")
    
    if args.dummy:
        adata._inplace_subset_obs(adata.obs_names[:100])
        print('using only first 100 cells')
    
    # Add genes for new gene set 
    existing_genes = adata.var_names.tolist()
    gene_set = list(set(list(meta_info['gene_set'])))
    new_genes = [g for g in gene_set if g not in existing_genes]
    print(f"Adding {len(new_genes)} new genes to adata")
    
    if len(new_genes) > 0:
        n_obs = adata.n_obs
        n_new_vars = len(new_genes)
        new_vars = pd.DataFrame(index=new_genes)
        new_var = pd.concat([adata.var, new_vars], axis=0)
    
        if sp.issparse(adata.X):
            new_data = sp.csr_matrix((n_obs, n_new_vars))
            X = sp.hstack([adata.X, new_data], format='csr')
        else:
            new_data = np.zeros((n_obs, n_new_vars))
            X = np.hstack([adata.X, new_data])
    
        adata = sc.AnnData(X=X, var=new_var, obs=adata.obs, obsm=adata.obsm, uns=adata.uns)
        adata.var_names_make_unique()
    
    # -----------------------------
    # NEW: version 1 — slice-based split if --val-slice-nums is provided
    # -----------------------------
    if args.val_slice_nums is not None:
        # Build slice name sets
        val_slice_names   = [f'{args.slice_prefix}{s}' for s in args.val_slice_nums]
        if args.slice_nums is not None:
            train_slice_names = [f'{args.slice_prefix}{s}' for s in args.slice_nums]
        else:
            train_slice_names = []  # use all other slices not in val
    
        # Compute union so we subset once, then do minimal copies
        union_names = sorted(set(train_slice_names) | set(val_slice_names))
        if len(union_names) > 0:
            mask_union = adata.obs['brain_section_label'].isin(union_names)
            adata._inplace_subset_obs(mask_union)
            print(f"Kept only union(train, val) slices → {adata}")
    
        # Now determine masks within this subset
        val_mask = adata.obs['brain_section_label'].isin(val_slice_names)
        if len(train_slice_names) > 0:
            train_mask = adata.obs['brain_section_label'].isin(train_slice_names)
        else:
            train_mask = ~val_mask  # everything else is train
    
        # Copy out val; in-place subset for train (minimize copies)
        adata_val = adata[val_mask].copy()
        adata._inplace_subset_obs(train_mask)
        adata_train = adata
    
        # If a second AnnData is provided, split it to match train/val counts and then concatenate
        if args.adata2 is not None:
            train_path = os.path.join(args.output_dir, "train.h5ad")

            if os.path.exists(train_path):
                print(f"Found existing {train_path}, loading instead of rebuilding...")
                adata_train = sc.read_h5ad(train_path)
            else:
                rng = np.random.default_rng(seed=getattr(args, "seed", None))
                adata2 = sc.read_h5ad(args.adata2)
                adata2.var_names_make_unique()
                adata2._inplace_subset_var(gene_set)
                if args.dummy:
                    adata2._inplace_subset_obs(adata2.obs_names[:100])
                    print('using only first 100 cells')
                
                print(adata2)

                # requested sizes
                n_val_req   = adata_val.n_obs
                n_train_req = adata_train.n_obs
                
                n_total = adata2.n_obs
                
                # 1) allocate what you can: val first, train from the remainder
                n_val   = min(n_val_req, n_total)
                n_train = min(n_train_req, n_total - n_val)
                
                # 2) sample indices (no replacement)
                all_idx = np.arange(n_total)
                idx_val = rng.choice(all_idx, size=n_val, replace=False) if n_val > 0 else np.array([], dtype=int)
                
                mask_val = np.zeros(n_total, dtype=bool)
                mask_val[idx_val] = True
                remain = np.flatnonzero(~mask_val)
                
                idx_train = rng.choice(remain, size=n_train, replace=False) if n_train > 0 else np.array([], dtype=int)
                
                # 3) build splits: make val as a separate object, keep train in-place
                adata2_val = adata2[idx_val].copy()          # new object for validation
                adata2._inplace_subset_obs(idx_train)        # in-place: adata2 becomes the train split
                adata2_train = adata2
                
                # (optional) housekeeping
                adata2_train.obs_names_make_unique()
                adata2_val.obs_names_make_unique()
                adata_train.obs_names_make_unique()
                
                print(f"Requested: train={n_train_req}, val={n_val_req}; "
                      f"Allocated: train={n_train}, val={n_val} (total available={n_total})")
                
                _set_uniform_sampling_prob(adata2_train, col='sampling_prob')
                
                # n_val   = adata_val.n_obs
                # n_train = adata_train.n_obs
                # if n_train > adata2.n_obs:
                #     raise ValueError(f"--adata2 has only {adata2.n_obs} cells, cannot split into "
                #                      f"{n_train} train + {n_val} val.")
        
                # # sample validation cells first (no replacement), then training from remaining
                # all_idx = np.arange(adata2.n_obs)
                # #idx_val = rng.choice(all_idx, size=n_val, replace=False)
                # #remain  = np.setdiff1d(all_idx, idx_val, assume_unique=False)
                # idx_train = rng.choice(all_idx, size=n_train, replace=False)
                # #adata2_val   = adata2[idx_val].copy()
                # adata2._inplace_subset_obs(idx_train)
                # adata2_train = adata2
                # print(adata2_train)
                # adata_train.obs_names_make_unique()
                # adata2_train.obs_names_make_unique()
                
        
                # Concatenate each split separately
                adata_train = sc.concat(
                    [adata_train,adata2_train], join='outer',
                    label='dataset_type', keys=['st', 'scrna']
                )
               
                # Ensure both sides have a weight column (ad2 got 1.0 earlier)
                rebalance_sampling_mass(
                    adata_train,
                    group_col='dataset_type',
                    sampling_col=args.sampling_col,          # 'sampling_prob'
                    target_fracs={'st': 0.7, 'scrna': 0.3}   # change if you want a different mix
                )
                print(adata_train)
                train_path = os.path.join(args.output_dir, "train.h5ad")
                if rank == 0:
                    adata_train.write(train_path)
            #if n_val > 0:
                #adata_val = adata_val.concatenate(
                    #adata2_val, join='outer',
                    #batch_key='dataset_type', batch_categories=['st', 'scrna']
                #)
    
    # -----------------------------
    # version 2 — random split if --val-slice-nums is NOT provided
    # (same behavior as before, but now we also save obs_names)
    # -----------------------------
    else:
        n_cells = adata.n_obs
        if args.val_split > 0:
            idxs = np.arange(n_cells)
            np.random.shuffle(idxs)
            n_val = int(n_cells * args.val_split)
            val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
    
            # Make a tiny, independent copy only for the validation set
            adata_val = adata[val_idxs].copy()
    
            # Train: inplace subset (no full copy)
            adata._inplace_subset_obs(~adata.obs_names.isin(adata.obs_names[val_idxs]))
            adata_train = adata
    
            # If a second AnnData is provided, match counts and concatenate like before
            if args.adata2 is not None:
                adata2 = sc.read_h5ad(args.adata2)
                num_adata_cells = len(adata_train.obs_names)# + len(adata_val.obs_names)
                if num_adata_cells > adata2.n_obs:
                    raise ValueError(f"adata2 only has {adata2.n_obs} cells, cannot match {num_adata_cells}")
    
                rng = np.random.default_rng(seed=getattr(args, "seed", None))
                # sample total, then split sampled into train/val sizes
                total_idx = rng.choice(adata2.n_obs, size=num_adata_cells, replace=False)
                n_train = adata_train.n_obs
                idx_train2 = total_idx[:n_train]
                #idx_val2   = total_idx[n_train:]
                adata2._inplace_subset_obs(idx_train2)
    
                adata2_train = adata2
                #adata2_val   = adata2[idx_val2].copy()
                _set_uniform_sampling_prob(adata2_train, col='sampling_prob')
                
                adata_train = adata_train.concatenate(
                    adata2_train, join='outer',
                    batch_key='dataset_type', batch_categories=['st', 'scrna']
                )
                #adata_val = adata_val.concatenate(
                    #adata2_val, join='outer',
                    #batch_key='dataset_type', batch_categories=['st', 'scrna']
                #)
    
            # Save obs_names so you can reuse them later
            train_obs_path = os.path.join(args.output_dir, "train_obs.txt")
            val_obs_path   = os.path.join(args.output_dir, "val_obs.txt")
            np.savetxt(train_obs_path, adata_train.obs_names.to_numpy(), fmt='%s')
            np.savetxt(val_obs_path,   adata_val.obs_names.to_numpy(),   fmt='%s')
            print(f"Saved train obs_names → {train_obs_path}")
            print(f"Saved val   obs_names → {val_obs_path}")
    
        else:
            adata_train = adata
            adata_val   = None
    
    args.use_sampling_probs = not args.disable_sampling_probs
    use_probs_val   = False  # usually evaluate uniformly
    
    # Optional: if you want to limit rows per epoch, pass args.epoch_samples if you have it
    epoch_samples = getattr(args, 'epoch_samples', None)
    base_seed     = getattr(args, 'seed', 0)
    
    train_loader = get_generation_dataloader(
        adata       = adata_train,
        meta_info   = meta_info,
        batch_size  = args.batch_size,
        max_len     = args.max_len,
        shuffle     = not args.no_shuffle,
        num_workers = args.num_workers,
        n_express_level = model.config.expression_level,
        include_0s  = False,
        add_xyz_noise = args.xyz_noise,
        # exclude_columns = ['supertype','cluster'],
        use_sampling_probs = args.use_sampling_probs,
        sampling_col       = args.sampling_col,
        epoch_samples      = epoch_samples,
        seed               = base_seed,
    )
    
    if adata_val is not None:
        val_loader = get_generation_dataloader(
            adata       = adata_val,
            meta_info   = meta_info,
            batch_size  = args.batch_size,
            max_len     = args.max_len,
            shuffle     = False,
            num_workers = args.num_workers,
            n_express_level = model.config.expression_level,
            include_0s  = False,
            # keep uniform sampling for validation
            use_sampling_probs = use_probs_val,
            sampling_col       = args.sampling_col
        )
    else:
        val_loader = None
    print('loaded anndata')

    # 5) Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.to(device)
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 6) Initialize W&B
    if rank == 0:
        wandb.init(
            project="scMulan-finetune",
            name=f'{os.path.basename(args.output_dir)}_bs{args.batch_size}',
            config={
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "lr":           args.lr,
                "lambda_val":   args.lambda_val,
                "val_split":    args.val_split,
            },
            dir=args.output_dir,
        )

    if args.from_finetuned:
        # e.g. ckp-path ends with ".../epoch3_model.pt"
        import re
        m = re.search(r'epoch(\d+)', os.path.basename(args.ckp_path))
        start_epoch = int(m.group(1)) + 1 if m else 1
    else:
        start_epoch = 1
        
    
    # 7) Training + validation loop
    for epoch in range(start_epoch, args.epochs+1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Grab the *local* indices from the sampler, then aggregate
        local_idx = None
        if hasattr(train_loader.sampler, "get_last_indices"):
            # Need to trigger one iterator to make __iter__ run at least once
            _ = iter(train_loader)
            local_idx = train_loader.sampler.get_last_indices()    
            summarize_sample_ddp(adata_train, local_idx, cell_key="cluster", group_key="dataset_type", top_k=20)





        
        model.train()
        total_loss, total_cls, total_exp_bin, total_exp_real = 0.0, 0.0, 0.0, 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [train]")):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch.get('labels')
            x_expr         = batch['input_vals'].to(device)
            expr_target    = batch['target_vals'].to(device)

            if labels is not None:
                labels = labels.to(device)

            optimizer.zero_grad()
            logits_cls, logits_exp_bins, logits_exp_real, loss, loss_cls, loss_exp_bin, loss_exp_real = model(
                idx=input_ids,
                x_expr=x_expr,
                targets=labels,
                y_expr=expr_target,
                lambda_val=args.lambda_val,
                return_hidden=False,
            )
            loss.backward()
            optimizer.step()

            total_loss   += loss.item() if loss is not None else 0.0
            total_cls    += loss_cls.item() if loss_cls is not None else 0.0
            total_exp_bin    += loss_exp_bin.item() if loss_exp_bin is not None else 0.0
            total_exp_real    += loss_exp_real.item() if loss_exp_real is not None else 0.0

            if rank == 0:
                if step % args.log_per_steps == 0:
                    wandb.log({
                        "train/batch_loss":     loss.item(),
                        "train/batch_loss_cls": loss_cls.item(),
                        "train/batch_loss_exp_bin": loss_exp_bin.item(),
                        "train/batch_loss_exp_real": loss_exp_real.item(),
                        "train/step":           (epoch-1)*len(train_loader) + step,
                    })

        avg_loss = total_loss / len(train_loader)
        avg_cls  = total_cls  / len(train_loader)
        avg_exp_bin  = total_exp_bin  / len(train_loader)
        avg_exp_real  = total_exp_real  / len(train_loader)
        print(f"Epoch {epoch} — train total {avg_loss:.4f}, cls {avg_cls:.4f}, exp_bin {avg_exp_bin:.4f}, exp_real {avg_exp_real:.4f}")
        if rank == 0:
            
            wandb.log({
                "train/epoch_loss":     avg_loss,
                "train/epoch_loss_cls": avg_cls,
                "train/epoch_loss_exp_bin": avg_exp_bin,
                "train/epoch_loss_exp_real": avg_exp_real,
                "epoch":                epoch,
            })

        if val_loader is not None:
            torch.cuda.empty_cache()
            model.eval()
            v_loss, v_cls, v_exp_bin, v_exp_real, count = 0.0, 0.0, 0.0, 0.0, 0
            all_pearson_rs = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    input_ids      = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels         = batch.get('labels')
                    x_expr         = batch['input_vals'].to(device)
                    expr_target    = batch['target_vals'].to(device)

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
                    v_cls  += lc.item()
                    v_exp_bin  += leb.item()
                    v_exp_real  += ler.item()
                    count  += 1
                    pearson_rs = evaluate_expression_correlation(
                        adata=adata_val,
                        batch=batch,
                        logits_labels=logits_labels.cpu(),
                        logits_exp_real=logits_exp_real.cpu(),
                        tokenizer=tokenizer,
                        tokens_and_vals_to_expression_row_fn=tokens_and_vals_to_expression_row,
                        var_names=adata_val.var_names.tolist(),
                        mask_token_id=tokenizer.pad_token_id
                    )
                    all_pearson_rs.extend(pearson_rs)

            avg_v_loss = v_loss / count
            avg_v_cls  = v_cls  / count
            avg_v_exp_bin  = v_exp_bin  / count
            avg_v_exp_real  = v_exp_real  / count
            print(f"Epoch {epoch} — valid total {avg_v_loss:.4f}, cls {avg_v_cls:.4f}, exp {avg_v_exp_bin:.4f}")
            mean_pearson_r = np.mean(all_pearson_rs)
            print(f"Validation mean Pearson r: {mean_pearson_r:.4f}")
            if rank == 0:
                wandb.log({
                    "valid/epoch_loss":     avg_v_loss,
                    "valid/epoch_loss_cls": avg_v_cls,
                    "valid/epoch_loss_exp_bin": avg_v_exp_bin,
                    "valid/epoch_loss_exp_real": avg_v_exp_real,
                    "valid/pearson_r": mean_pearson_r,
                    "epoch":                epoch,
                })

        # Save epoch checkpoint
        if epoch % args.save_frequency == 0 and rank == 0:
            ckpt_file = os.path.join(args.output_dir, f"epoch{epoch}_model.pt")
            if dist.is_available() and dist.is_initialized():
                torch.save({'model': model.module.state_dict(),
                            'model_args': ckp['model_args']}, ckpt_file)
            else:
                torch.save({'model': model.state_dict(),
                            'model_args': ckp['model_args']}, ckpt_file)

    # 8) Save final artifacts
    # model.save_pretrained(args.output_dir)
    MulanConfig(**ckp['model_args']).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Finetuned artifacts written to {args.output_dir}")
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
