from anndata import AnnData
import torch
from torch.utils.data import Dataset, DataLoader
from utils.hf_tokenizer import scMulanTokenizer
from scMulan import scMulan, fine_tuning, generate_prompt_for_cg
import inspect
from typing import Tuple, List
# from scMulan import generate_cellGenesis

import pandas as pd
import numpy as np



class _AnndataGenDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        meta_pool: dict,
        sep_token: str,
        tokenizer: scMulanTokenizer,
        max_len: int,
        scm: scMulan,
        include_0s: bool,
    ):
        """
        Expects:
          - adata.obs has whatever columns generate_prompt_for_cg() needs
          - adata.obs['gene_tokens'] is a list of gene‐token strings per cell
        """
        self.adata     = adata
        self.meta_pool = meta_pool
        self.sep_token = sep_token
        self.tok       = tokenizer
        self.max_len   = max_len
        self.scm   = scm
        self.include_0s = include_0s

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        # 1) build the prompt
        prompt_tokens, prompt_vals = generate_prompt_for_cg(idx, self.adata.obs, self.meta_pool, self.tok)
        prompt_tokens += self.tok.encode([self.sep_token])
        prompt_vals += [0]
 
        target_tokens, target_vals, target_real_vals = self.scm.get_gene_and_expression_tokens(idx, self.include_0s)
        # 3) full token sequence
        ids = prompt_tokens + target_tokens + self.tok.encode(['<E>'])
        vals = prompt_vals + list(target_vals) + [0]
        target_real_vals = prompt_vals + list(target_real_vals) + [0]


        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            vals = vals[:self.max_len]
        pad_len = self.max_len - len(ids)
        ids += [self.tok.pad_token_id] * pad_len
        vals += [0] * pad_len
        target_real_vals += [0] * pad_len

        # 5) labels: ignore prompt
        labels = [-100] * len(prompt_tokens) + target_tokens + self.tok.encode(['<E>'])
        if len(labels) > self.max_len:
            labels = labels[: self.max_len]
        labels += [-100] * (self.max_len - len(labels))
        # ids += [-100] * (self.max_len - len(ids))
        # vals += [-100] * (self.max_len - len(vals))

        # 6) attention mask
        mask = [1 if i != self.tok.pad_token_id else 0 for i in ids]


        input_ids   = torch.tensor(ids,   dtype=torch.long).clone()
        labels      = torch.tensor(labels,dtype=torch.long).clone()
        attention_mask = torch.tensor(mask,  dtype=torch.long).clone()
        input_vals  = torch.tensor(vals,  dtype=torch.long).clone()
        target_vals = torch.tensor(target_real_vals, dtype=torch.float32).clone()
        # print(input_ids.shape, labels.shape, attention_mask.shape, input_vals.shape, target_vals.shape)
    
        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
            "input_vals":     input_vals,
            "target_vals":    target_vals,
        }

def get_generation_dataloader(
    adata: AnnData,
    meta_info: dict,
    batch_size: int = 8,
    max_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    include_0s: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for fine‑tuning on the conditional‑generation task.
    
    Parameters
    ----------
    adata : AnnData
        Your AnnData with `.obs` containing both the fields that
        `generate_prompt_for_cg` needs and a column
        `adata.obs['gene_tokens']`, where each entry is a list of
        gene‐token strings for that cell.
    meta_info : dict
        Loaded from your `meta_info.pt`. Must contain:
          - 'token_set'   : list of all tokens
          - 'sep_token'   : e.g. '<SPToken1>'
          - 'meta_pool'   : whatever generate_prompt_for_cg expects
    batch_size, max_len, shuffle, num_workers : as usual
    
    Returns
    -------
    DataLoader yielding dicts with keys 'input_ids', 'labels', 'attention_mask', 'expression'
    """
    # 1) tokenizer
    tokenizer = scMulanTokenizer(meta_info['token_set'])
    tokenizer.add_special_tokens({'sep_token': meta_info.get('sep_token', '<SPToken1>')})
    tokenizer.pad_token = '<SPToken10>'

    scm = fine_tuning(adata, meta_info)
    scm.data_preprocess()

    # 2) dataset
    ds = _AnndataGenDataset(
        adata,
        meta_pool = meta_info,
        sep_token = meta_info.get('sep_token', '<SPToken1>'),
        tokenizer = tokenizer,
        max_len   = max_len,
        scm   = scm,
        include_0s = include_0s,
    )

    # 3) dataloader
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
    )


