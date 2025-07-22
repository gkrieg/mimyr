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

root_path = os.path.abspath('/work/magroup/skrieger/scMulan/Tutorials/scMulan')
sys.path.append(os.path.abspath(root_path))

from model.model import MulanConfig, scMulanModel
from utils.hf_tokenizer import scMulanTokenizer
from data_util import get_generation_dataloader

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune scMulanModel on conditional generation with coords + validation"
    )
    parser.add_argument("--ckp-path",    type=str, required=True,
                        help="Path to pretrained checkpoint (.pt)")
    parser.add_argument("--meta-info",   type=str, required=True,
                        help="Path to meta_info.pt from pretraining")
    parser.add_argument("--adata",       type=str, required=True,
                        help="Path to input AnnData .h5ad file")
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
    parser.add_argument("--slice-nums", nargs='+',type=int, default=None,
                        help='List of slice IDs for spatial mouse brain atlases')
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
    ckp = torch.load(args.ckp_path, map_location='cpu')
    if args.overwrite_vocab_size is not None:
        ckp['model_args']['vocab_size'] = args.overwrite_vocab_size
        print(f"Overwriting config.vocab_size to {args.overwrite_vocab_size}")
    if args.new_expression_size is not None and args.from_finetuned:
        ckp['model_args']['expression_level'] = args.new_expression_size
    gptconf = MulanConfig(**ckp['model_args'])
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
    # 4) Load AnnData and split into train/validation
    adata = sc.read_h5ad(args.adata)

    if args.slice_nums is not None:
        slice_names = [f'C57BL6J-638850.{s}' for s in args.slice_nums]
        obs_in_slices = adata.obs_names[adata.obs['brain_section_label'].isin(slice_names)]
        adata._inplace_subset_obs(adata.obs['brain_section_label'].isin(slice_names))
        print(f'Subsetted adata to only contain {args.slice_nums} slices giving {adata}')

    # Add genes for mulan gene set
    existing_genes = adata.var_names.tolist()
    
    gene_set = list(meta_info['gene_set'])
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

        new_adata = sc.AnnData(X=X, var=new_var, obs=adata.obs, obsm=adata.obsm, uns=adata.uns)
        new_adata.var_names_make_unique()
        del adata
        adata = new_adata

    
    print('adata loaded')
    n_cells = adata.n_obs
    if args.val_split > 0:
        idxs = np.arange(n_cells)
        np.random.shuffle(idxs)
        n_val = int(n_cells * args.val_split)
        val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
        adata_val = adata[val_idxs].copy()  # Small, independent copy
        adata._inplace_subset_obs(~adata.obs_names.isin(adata.obs_names[val_idxs]))
        adata_train = adata
    else:
        adata_train = adata
        adata_val   = None

    train_loader = get_generation_dataloader(
        adata      = adata_train,
        meta_info  = meta_info,
        batch_size = args.batch_size,
        max_len    = args.max_len,
        shuffle    = not args.no_shuffle,
        num_workers= args.num_workers,
        n_express_level=model.config.expression_level,
        include_0s = False,
        add_xyz_noise = args.xyz_noise,
    )
    if adata_val is not None:
        val_loader = get_generation_dataloader(
            adata      = adata_val,
            meta_info  = meta_info,
            batch_size = args.batch_size,
            max_len    = args.max_len,
            shuffle    = False,
            num_workers= args.num_workers,
            n_express_level=model.config.expression_level,
            include_0s = False,
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
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    input_ids      = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels         = batch.get('labels')
                    x_expr         = batch['input_vals'].to(device)
                    expr_target    = batch['target_vals'].to(device)

                    if labels is not None:
                        labels = labels.to(device)

                    _, _, _, l, lc, leb, ler = model(
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

            avg_v_loss = v_loss / count
            avg_v_cls  = v_cls  / count
            avg_v_exp_bin  = v_exp_bin  / count
            avg_v_exp_real  = v_exp_real  / count
            print(f"Epoch {epoch} — valid total {avg_v_loss:.4f}, cls {avg_v_cls:.4f}, exp {avg_v_exp_bin:.4f}")
            if rank == 0:
                wandb.log({
                    "valid/epoch_loss":     avg_v_loss,
                    "valid/epoch_loss_cls": avg_v_cls,
                    "valid/epoch_loss_exp_bin": avg_v_exp_bin,
                    "valid/epoch_loss_exp_real": avg_v_exp_real,
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
