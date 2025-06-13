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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load pretrained checkpoint
    ckp = torch.load(args.ckp_path, map_location='cpu')
    if args.overwrite_vocab_size is not None:
        ckp['model_args']['vocab_size'] = args.overwrite_vocab_size
        print(f"Overwriting config.vocab_size to {args.overwrite_vocab_size}")
    gptconf = MulanConfig(**ckp['model_args'])
    ModelClass = scMulanModel
    model = ModelClass(gptconf)
    device = torch.device(args.device)
    model.to(device)
    model.load_state_dict(ckp['model'], strict=False)
    model.eval()
    model.hidden_dim = ckp['model_args']['n_embd']

    # 2) Load and extend meta_info with coordinate tokens
    meta_info = torch.load(args.meta_info)
    new_tokens = ["<x>", "<y>", "<z>"]
    meta_info['token_set'].extend(new_tokens)

    print('loaded meta_info')
    
    # 3) Initialize tokenizer and resize model embeddings/output
    tokenizer = scMulanTokenizer(meta_info['token_set'])
    if not args.from_finetuned:
        sep = meta_info.get('sep_token', '<SPToken1>')
        tokenizer.add_special_tokens({'sep_token': sep})
        # resize_token_embeddings comes from PreTrainedModel
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
    model.to(device)

    print('initialized model')
    # 4) Load AnnData and split into train/validation
    adata = sc.read_h5ad(args.adata)
    n_cells = adata.n_obs
    if args.val_split > 0:
        idxs = np.arange(n_cells)
        np.random.shuffle(idxs)
        n_val = int(n_cells * args.val_split)
        val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
        adata_train = adata[train_idxs].copy()
        adata_val   = adata[val_idxs].copy()
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
        include_0s = False,
    )
    if adata_val is not None:
        val_loader = get_generation_dataloader(
            adata      = adata_val,
            meta_info  = meta_info,
            batch_size = args.batch_size,
            max_len    = args.max_len,
            shuffle    = False,
            num_workers= args.num_workers,
            include_0s = False,
        )
    else:
        val_loader = None
    print('loaded anndata')
    # 5) Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 6) Initialize W&B
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
        model.train()
        total_loss, total_cls, total_exp = 0.0, 0.0, 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [train]")):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch.get('labels')
            x_expr         = batch['input_vals'].to(device)
            expr_target    = batch['target_vals'].to(device)

            if labels is not None:
                labels = labels.to(device)

            optimizer.zero_grad()
            logits_cls, logits_exp, loss, loss_cls, loss_exp = model(
                idx=input_ids,
                x_expr=x_expr,
                targets=labels,
                y_expr=expr_target,
                lambda_val=args.lambda_val,
                return_hidden=False,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls  += loss_cls.item()
            total_exp  += loss_exp.item()

            wandb.log({
                "train/batch_loss":     loss.item(),
                "train/batch_loss_cls": loss_cls.item(),
                "train/batch_loss_exp": loss_exp.item(),
                "train/step":           (epoch-1)*len(train_loader) + step,
            })

        avg_loss = total_loss / len(train_loader)
        avg_cls  = total_cls  / len(train_loader)
        avg_exp  = total_exp  / len(train_loader)
        print(f"Epoch {epoch} — train total {avg_loss:.4f}, cls {avg_cls:.4f}, exp {avg_exp:.4f}")
        wandb.log({
            "train/epoch_loss":     avg_loss,
            "train/epoch_loss_cls": avg_cls,
            "train/epoch_loss_exp": avg_exp,
            "epoch":                epoch,
        })

        if val_loader is not None:
            model.eval()
            v_loss, v_cls, v_exp, count = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    input_ids      = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels         = batch.get('labels')
                    x_expr         = batch['input_vals'].to(device)
                    expr_target    = batch['target_vals'].to(device)

                    if labels is not None:
                        labels = labels.to(device)

                    _, _, l, lc, le = model(
                        idx=input_ids,
                        x_expr=x_expr,
                        targets=labels,
                        y_expr=expr_target,
                        lambda_val=args.lambda_val,
                        return_hidden=False,
                    )
                    v_loss += l.item()
                    v_cls  += lc.item()
                    v_exp  += le.item()
                    count  += 1

            avg_v_loss = v_loss / count
            avg_v_cls  = v_cls  / count
            avg_v_exp  = v_exp  / count
            print(f"Epoch {epoch} — valid total {avg_v_loss:.4f}, cls {avg_v_cls:.4f}, exp {avg_v_exp:.4f}")
            wandb.log({
                "valid/epoch_loss":     avg_v_loss,
                "valid/epoch_loss_cls": avg_v_cls,
                "valid/epoch_loss_exp": avg_v_exp,
                "epoch":                epoch,
            })

        # Save epoch checkpoint
        if epoch % args.save_frequency == 0:
            ckpt_file = os.path.join(args.output_dir, f"epoch{epoch}_model.pt")
            torch.save({'model': model.state_dict(),
                        'model_args': ckp['model_args']}, ckpt_file)

    # 8) Save final artifacts
    # model.save_pretrained(args.output_dir)
    MulanConfig(**ckp['model_args']).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Finetuned artifacts written to {args.output_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()
