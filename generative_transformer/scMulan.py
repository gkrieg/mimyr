import torch
import os
import sys
from .model.model import MulanConfig, scMulanModel
from .model.model_kvcache import scMulanModel_kv
import torch.nn.functional as F
from .utils.hf_tokenizer import scMulanTokenizer
import scipy.sparse
import numpy as np
from tqdm import tqdm
from anndata import AnnData
from typing import Optional, List, Tuple, Union
import pandas as pd
import io
import multiprocessing
from tqdm import tqdm
import time
multiprocessing.set_start_method('spawn',force=True)

class scMulan:
    # def __init__(self, model, adata, meta_info, tokenizer, n_express_level, **kwargs):
    #     self.model = model
    #     self.meta_info = meta_info
    #     self.tokenizer = tokenizer
    #     self.mulan_gene_set = self.meta_info['gene_set']
    #     self.n_express_level = n_express_level
    #     self.check_adata(adata,**kwargs)
    #     self.mulan_cell_type_entities = list(self.meta_info['cell_type'] | self.meta_info['MCT'])
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def __init__(self, adata, meta_info, tokenizer, n_express_level):
    #     self.meta_info = meta_info
    #     self.tokenizer = tokenizer
    #     self.mulan_gene_set = self.meta_info['gene_set']
    #     self.check_adata(adata)
    #     self.n_express_level = n_express_level
    #     self.mulan_cell_type_entities = list(self.meta_info['cell_type'] | self.meta_info['MCT'])
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        adata: AnnData,
        meta_info: dict,
        tokenizer,
        n_express_level: int,
        bin_edges: Optional[np.ndarray] = None,
        model: Optional[scMulanModel] = None,
        **kwargs
    ):
        if model is not None:
            self.model  = model
        self.adata  = adata
        self.meta_info = meta_info
        self.tokenizer = tokenizer
        self.n_express_level = n_express_level
        if bin_edges is not None:
            self.bin_edges = torch.tensor(bin_edges)

        self.mulan_gene_set = meta_info['gene_set']
        # self.mulan_cell_type_entities = list(meta_info['cell_type'] | meta_info['MCT'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.check_adata(adata, **kwargs)

    def data_preprocess(self,):

        # sparse check
        # self.adata_sparse = False #scipy.sparse.issparse(self.adata.X) # TODO use sparsity
        # # get COO matrix for analysis
        # if self.adata_sparse:
        #     self.adata_matrix = self.adata.X.tocoo()
        # else:pass
            #print('adata is not sparse, use dense matrix and dataframe')
            # self.adata_matrix = self.adata.X.toarray()
        cellDFHVG = pd.DataFrame(self.adata.X.toarray(), columns = self.mulan_gene_set)
        cellDFHVG.index = list(self.adata.obs.index)
        self.adata_matrix = cellDFHVG

        


    def get_gene_expression_dict(self, i, matrix):
        genes_series = matrix.iloc[i]
        expressed_genes = genes_series[genes_series > 0].index.tolist()
        expr_values = genes_series[expressed_genes].values
        cell_expression_dict = {gene: expr_value for gene, expr_value in zip(expressed_genes, expr_values)}
        return cell_expression_dict

    def get_gene_expression_dict_with_0s(self, i, matrix):
        genes_series = matrix.iloc[i]
        expressed_genes = genes_series.index.tolist()
        expr_values = genes_series.values
        cell_expression_dict = {gene: expr_value for gene, expr_value in zip(expressed_genes, expr_values)}
        return cell_expression_dict

    def prepare_gene_expression_codings(self, i, matrix, include_0s=False):
        # 1) build the dict of gene→expr, possibly including zeros
        if include_0s:
            cell_expression_dict = self.get_gene_expression_dict_with_0s(i, matrix)
        else:
            cell_expression_dict = self.get_gene_expression_dict(i, matrix)
    
        # preserve your original ordering
        expressed_genes   = list(cell_expression_dict.keys())[::-1]
        expression_values = np.array(list(cell_expression_dict.values())[::-1], dtype=float)
    
        # 2) identify zeros and positives
        zero_mask     = (expression_values == 0)
        nonzero_vals  = expression_values[~zero_mask]
    
        # 3) compute bins *only* on the positive values
        if len(nonzero_vals) > 0:
            # digitize positives into 1…N
            pos_bins = np.digitize(nonzero_vals, self.bin_edges, right=True)
        else:
            pos_bins = np.array([], dtype=int)
    
        # 4) merge back into a full-length array, with zeros→bin 0
        binned_expr = np.empty_like(expression_values, dtype=int)
        # fill zeros
        binned_expr[zero_mask] = 0
        # fill positives
        binned_expr[~zero_mask] = pos_bins
        # print(len(expressed_genes), binned_expr.shape, expression_values.shape)
        return expressed_genes, binned_expr, expression_values

    
    # def prepare_gene_expression_codings(self, i, matrix, include_0s=False):

    #     if include_0s == True:
    #         cell_expression_dict = self.get_gene_expression_dict_with_0s(i, matrix)
    #     else:
    #         cell_expression_dict = self.get_gene_expression_dict(i, matrix)
    #     expressed_genes = list(cell_expression_dict.keys())[::-1]
    #     expression_values = list(cell_expression_dict.values())[::-1]
    #     max_expression = np.max(expression_values)
    #     bins = np.linspace(0, max_expression, self.n_express_level+1)
    #     binned_expr = np.digitize(expression_values, bins, right=True)

    #     return expressed_genes, binned_expr
    
    def make_encoded_annotation_prompt_one_cell(self, expressed_genes, binned_expr, annotation_task_token = '<PCT>'):

        prefix = expressed_genes + [annotation_task_token] # add pre-defined task token to guide model generate cell type annotations
        ec_binned_expr = np.append(binned_expr,[0]*(len([annotation_task_token]))) # add a zero for task token
        ec_prefix = self.tokenizer.encode(prefix) 
        prefix_len_with_task_token = len(ec_prefix) # length with task token

        return (ec_prefix, ec_binned_expr, prefix_len_with_task_token)

    def make_encoded_gene_expression_one_cell(self, expressed_genes, binned_expr):

        prefix = expressed_genes 
        ec_binned_expr = binned_expr
        ec_prefix = self.tokenizer.encode(prefix) 

        return (ec_prefix, ec_binned_expr)
    

    def get_cell_type(self, i, matrix, **kwargs):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
            ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
            prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0).to(device)
            prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0).to(device)
            generated_tokens = self.model.generate_cellGenesis(prompt_entities,prompt_values, max_new_tokens= prefix_len_with_task_token + 3, top_k=1, **kwargs)[0].cpu().tolist()
            pred_names = self.tokenizer.convert_ids_to_tokens(generated_tokens[0][-3:-1])
            coarse_cell_type = pred_names[-2] if self.is_cell_type_entity(pred_names[-2]) else 'Unclassified'
            fine_cell_type = pred_names[-1] if self.is_cell_type_entity(pred_names[-1]) else 'Unclassified'

        return coarse_cell_type, fine_cell_type
    
    def cell_type_pred_process_subdata(self, idx_subset, device_id, save_path = None, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')

        self.model.to(device)
        fine_cell_type_pred = []
        # print(f'using device {device_id}, on processing {os.getpid()}')
        for idx in tqdm(idx_subset, desc=f"⏳Generating cell type labels for each cell on device {device_id}"):
            coarse_cell_type, fine_cell_type = self.get_cell_type(idx, self.adata_matrix, **kwargs)
            fine_cell_type_pred.append(fine_cell_type)
        torch.cuda.empty_cache()
        
        return fine_cell_type_pred

    def get_cell_types_for_adata(self, parallel = False, n_process = None,  **kwargs):

        self.data_preprocess()
        if parallel:
            assert n_process is not None, print('n_process must be set if using parallel')
            print(f'⚡ Speed up by multiprocessing with {n_process} processes and {torch.cuda.device_count()} GPUs...')
            fine_cell_type_pred = self.process_data_in_parallel(self.cell_type_pred_process_subdata, n_process)
            self.adata.obs['cell_type_from_scMulan'] = fine_cell_type_pred

        fine_cell_type_pred = []
        for i in tqdm(range(self.adata.n_obs), desc="⏳Generating cell type labels for each cell"):
            _, fine_cell_type = self.get_cell_type(i, self.adata_matrix, **kwargs)
            fine_cell_type_pred.append(fine_cell_type)
        self.adata.obs['cell_type_from_scMulan'] = fine_cell_type_pred

    
    def get_cell_embedding(self, i, matrix, **kwargs):

        with torch.no_grad():
            expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
            ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
            prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0).cuda()
            prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0).cuda()
            _,_,hidden = self.model.generate_cellGenesis(prompt_entities,prompt_values,
                                                                    max_new_tokens= prefix_len_with_task_token + 3,
                                                                    top_k=1, return_hidden=True,**kwargs) # +3 is passing CT1, CT2,<#E#>
            hidden = hidden[-1][0,-2,:].cpu().numpy() #TODO custom choose embedding

        return hidden
    
    def embedding_process_subdata(self, idx_subset, device_id, save_path = None, **kwargs):

        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
        self.model.to(device)

        hidden_states = np.zeros((len(idx_subset), self.model.hidden_dim))

        for j,idx in enumerate(tqdm(idx_subset, desc=f"⏳ Collecting cell embeddings for each cell on device {device_id}")):
            hidden = self.get_cell_embedding(idx, self.adata_matrix, **kwargs)
            hidden_states[j] = hidden

        torch.cuda.empty_cache()
        if save_path:
            torch.save(hidden_states, save_path)

        return hidden_states
    
    def get_cell_embeddings_for_adata(self, parallel = False, n_process = None, save_dir = None, **kwargs):

        self.data_preprocess()
        if parallel:
            assert n_process is not None, print('n_process must be set if using parallel')
            print(f'⚡ Speed up by multiprocessing with {n_process} processes and {torch.cuda.device_count()} GPUs...')
            # hidden_states = self.process_data_in_parallel(self.embedding_process_subdata, n_process, save_dir)
            hidden_states = self.process_data_in_parallel(self.embedding_process_subdata, n_process, save_dir)
        else:
            hidden_states = []
            for i in tqdm(range(self.adata.n_obs), desc="⏳Collecting cell embeddings for each cell"):
                hidden = self.get_cell_embedding(i, self.adata_matrix, **kwargs)
                hidden_states.append(hidden)
        self.adata.obsm['X_scMulan'] = np.array(hidden_states)
    

    def get_cell_type_and_embd(self, i, matrix, **kwargs):

        with torch.no_grad():
            expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
            ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
            prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0)
            prompt_entities = prompt_entities.cuda() if torch.cuda.is_available() else prompt_entities
            prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0)
            prompt_values = prompt_values.cuda() if torch.cuda.is_available() else prompt_values
            generated_entities, generated_values, hidden = self.model.generate_cellGenesis(prompt_entities,prompt_values, 
                                                                                            max_new_tokens= prefix_len_with_task_token + 3,
                                                                                            top_k=1, return_hidden=True, **kwargs) # +3 is passing CT1, CT2,<#E#>
            pred_names = self.tokenizer.convert_ids_to_tokens(generated_entities[0].cpu().tolist()[-3:-1])
            # coarse_cell_type = pred_names[-2] if self.is_cell_type_entity(pred_names[-2]) else 'Unclassified'
            fine_cell_type = pred_names[-1] if self.is_cell_type_entity(pred_names[-1]) else 'Unclassified'
            hidden = hidden[-1][0,-2,:].cpu().numpy()

        return fine_cell_type, hidden


    def get_gene_and_expression_tokens(self, i, include_0s=True):
        
        expressed_genes, binned_expr, expression_vals = self.prepare_gene_expression_codings(i, self.adata_matrix, include_0s=include_0s)
        gene_tokens, expression_tokens = self.make_encoded_gene_expression_one_cell(expressed_genes, binned_expr)
        
        return (gene_tokens, expression_tokens, expression_vals)


    def cell_type_and_embd_process_subdata(self, idx_subset, device_id, save_path = None, **kwargs):        
        torch.cuda.set_device(device_id)
        self.model.to(device_id)
        pred_embd_list = []

        for idx in tqdm(idx_subset, desc=f"⏳ Generating cell type labels and embds for each cell on device {device_id}"):
            # fine_cell_type, hidden = self.get_cell_type_and_embd(idx, self.adata_matrix,**kwargs)
            fine_cell_type, hidden = self.get_cell_type_and_embd(idx, self.adata_matrix, **kwargs)
            pred_embd_list.append([fine_cell_type, hidden])

        torch.cuda.empty_cache()
        if save_path:
            torch.save(pred_embd_list, save_path)

        return pred_embd_list
    
    
    def get_cell_types_and_embds_for_adata(self, parallel = False, n_process = None, save_dir = None, **kwargs):

        self.data_preprocess()
        if parallel:
            assert n_process is not None, print('n_process must be set if using parallel')
            print(f'⚡ Speed up by multiprocessing with {n_process} processes and {torch.cuda.device_count()} GPUs...')
            results = self.process_data_in_parallel(self.cell_type_and_embd_process_subdata, n_process, save_dir)
        else:
            results = []
            for idx in tqdm(self.adata.obs_names, desc="⏳ Collecting cell embeddings for each cell"):
                ct, hidden = self.get_cell_type_and_embd(idx, self.adata_matrix, **kwargs)
                results.append([ct, hidden])

        cell_types = [pair[0] for pair in results]
        hidden_embds = [pair[1] for pair in results]
        self.adata.obs['cell_type_from_scMulan'] = cell_types
        self.adata.obsm['X_scMulan'] = np.array(hidden_embds)

    # @torch.no_grad()
    # def generate_cell_genesis(
    #     self,
    #     idx: int,
    #     max_new_tokens: int = 50,
    #     top_k: int = 5,
    #     return_gt: bool = False,
    #     cheat_with_tokens: bool = False,
    #     cheat_with_expr: bool = False,
    #     **generate_kwargs
    # ) -> List[str]:
    #     """
    #     Generate a gene‐token sequence for cell `idx` via the model’s
    #     generate_cellGenesis API.

    #     Returns
    #     -------
    #     List[str]
    #         The list of predicted gene‐token strings.
    #     """
    #     self.data_preprocess()
    #     # 1) Build the prompt IDs + values
    #     prompt_ids, prompt_vals = generate_prompt_for_cg(
    #         idx,
    #         self.adata.obs,
    #         self.meta_info,
    #         self.tokenizer
    #     )
    #     # 2) Append separator token and a dummy zero‐value
    #     sep_id = self.tokenizer.convert_tokens_to_ids(self.meta_info.get('sep_token', '<SPToken1>'))
    #     prompt_ids.append(sep_id)
    #     prompt_vals.append(0)
        
    #     if return_gt == True or cheat_with_tokens == True or cheat_with_expr == True:
    #         target_tokens, target_vals, target_real_vals = self.get_gene_and_expression_tokens(idx, include_0s=False)
    #         target_tokens = prompt_ids + target_tokens + self.tokenizer.encode(['<E>'])
    #         target_vals = prompt_vals + list(target_vals) + [0]
    #         target_real_vals = prompt_vals + list(target_real_vals) + [0]
    #         target_tokens   = torch.tensor(target_tokens,   dtype=torch.long).clone().unsqueeze(0).to(self.device)
    #         target_vals   = torch.tensor(target_vals,   dtype=torch.long).clone().unsqueeze(0).to(self.device)
    #         target_real_vals   = torch.tensor(target_real_vals,   dtype=torch.float32).clone().unsqueeze(0).to(self.device)

    #     # 3) Convert to tensors, batch dim = 1
    #     input_ids   = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
    #     input_vals  = torch.tensor([prompt_vals], dtype=torch.long, device=self.device)

    #     # 4) Call the model’s generate_cellGenesis
    #     generated_ids, generated_vals_binned, generated_vals = self.model.generate_cellGenesis(
    #         input_ids = input_ids,
    #         expression_level   = input_vals,
    #         max_new_tokens  = max_new_tokens + input_ids.shape[1],
    #         top_k           = top_k,
    #         override_gene_sequence = None if cheat_with_tokens != True else target_tokens,
    #         override_expr_sequence = None if cheat_with_expr != True else target_vals,
    #         **generate_kwargs
    #     )

    #     # generated_ids: Tensor shape (batch=1, total_len)
    #     gen_seq = generated_ids[0].tolist()  # drop batch dim
    #     gen_vals_binned = generated_vals_binned[0].tolist()
    #     gen_vals = generated_vals[0].tolist()

        

    #     # 5) Remove the prompt prefix and sep, return only the newly generated gene tokens
    #     n_prefix = len(prompt_ids)
    #     new_ids   = gen_seq[n_prefix:]
    #     new_vals  = gen_vals[n_prefix:]

    #     # 6) Convert IDs → token strings
    #     gene_tokens = self.tokenizer.convert_ids_to_tokens(new_ids)

    #     row = self.tokens_and_vals_to_expression_row(
    #         var_names    = self.adata.var_names.tolist(),
    #         gene_tokens  = gene_tokens,
    #         new_vals     = new_vals,
    #         return_series=False
    #     )
    #     if return_gt == True:
    #         return row, gene_tokens, new_vals, gen_seq, gen_vals_binned, gen_vals, target_tokens, target_vals, target_real_vals
    #     return row, gene_tokens, new_vals, gen_seq, gen_vals_binned, gen_vals

    

    @torch.no_grad()
    def generate_cell_genesis(
        self,
        idx: Union[int, List[int]],
        max_new_tokens: int = 50,
        top_k: int = 5,
        return_gt: bool = False,
        cheat_with_tokens: bool = False,
        cheat_with_expr: bool = False,
        batch_size: int = 12,
        verbose: bool = False,
        **generate_kwargs
    ):
        """
        Generate gene-token sequences for idx via mini-batches of generate_cellGenesis calls.
    
        Parameters
        ----------
        idx : int or List[int]
            Single index or list of indices from adata.obs
        batch_size : int
            Number of cells to process per model call
        verbose : bool
            If True, prints timing for each major step
    
        Returns
        -------
        List of generated results per cell
        """
        self.data_preprocess()
    
        if isinstance(idx, int):
            idx = [idx]
    
        results = []
        for start in tqdm(range(0, len(idx), batch_size)):
            batch_start_time = time.time()
            batch_indices = idx[start : start + batch_size]
    
            if verbose:
                print(f"\nProcessing batch {start // batch_size + 1} / {((len(idx) - 1) // batch_size) + 1}")
    
            # === Prompt Construction ===
            t0 = time.time()
            prompt_ids_list = []
            prompt_vals_list = []
            target_tokens_list = []
            target_vals_list = []
            target_real_vals_list = []
    
            sep_id = self.tokenizer.convert_tokens_to_ids(self.meta_info.get('sep_token', '<SPToken1>'))
    
            for i in batch_indices:
                prompt_ids, prompt_vals = generate_prompt_for_cg(
                    i, self.adata.obs, self.meta_info, self.tokenizer
                )
                prompt_ids.append(sep_id)
                prompt_vals.append(0)
    
                prompt_ids_list.append(prompt_ids)
                prompt_vals_list.append(prompt_vals)
    
            if verbose:
                print(f"  Prompt construction took {time.time() - t0:.2f} sec")
    
            # === Target Sequence Construction (optional) ===
            if return_gt or cheat_with_tokens or cheat_with_expr:
                t1 = time.time()
                for i, prompt_ids, prompt_vals in zip(batch_indices, prompt_ids_list, prompt_vals_list):
                    tgt_tokens, tgt_vals, tgt_real_vals = self.get_gene_and_expression_tokens(i, include_0s=False)
                    tgt_tokens = prompt_ids + tgt_tokens + self.tokenizer.encode(['<E>'])
                    tgt_vals = prompt_vals + list(tgt_vals) + [0]
                    tgt_real_vals = prompt_vals + list(tgt_real_vals) + [0]
    
                    target_tokens_list.append(tgt_tokens)
                    target_vals_list.append(tgt_vals)
                    target_real_vals_list.append(tgt_real_vals)
                if verbose:
                    print(f"  Target sequence creation took {time.time() - t1:.2f} sec")
    
            # === Tensor Preparation ===
            t2 = time.time()
            max_len = max(len(p) for p in prompt_ids_list)
            pad_token = 0
    
            input_ids = torch.full((len(batch_indices), max_len), pad_token, dtype=torch.long, device=self.device)
            input_vals = torch.full((len(batch_indices), max_len), pad_token, dtype=torch.long, device=self.device)
    
            for b, (p_ids, p_vals) in enumerate(zip(prompt_ids_list, prompt_vals_list)):
                plen = len(p_ids)
                input_ids[b, :plen] = torch.tensor(p_ids, dtype=torch.long, device=self.device)
                input_vals[b, :plen] = torch.tensor(p_vals, dtype=torch.long, device=self.device)
    
            override_gene_sequence = None
            override_expr_sequence = None
            if return_gt or cheat_with_tokens or cheat_with_expr:
                max_target_len = max(len(t) for t in target_tokens_list)
                override_gene_sequence = torch.full((len(batch_indices), max_target_len), pad_token, dtype=torch.long, device=self.device)
                override_expr_sequence = torch.full((len(batch_indices), max_target_len), pad_token, dtype=torch.long, device=self.device)
    
                for b, (t_ids, t_vals) in enumerate(zip(target_tokens_list, target_vals_list)):
                    tlen = len(t_ids)
                    override_gene_sequence[b, :tlen] = torch.tensor(t_ids, dtype=torch.long, device=self.device)
                    override_expr_sequence[b, :tlen] = torch.tensor(t_vals, dtype=torch.long, device=self.device)
    
            if verbose:
                print(f"  Tensor preparation took {time.time() - t2:.2f} sec")
    
            # === Model Call ===
            t3 = time.time()
            generated_ids, generated_vals_binned, generated_vals = self.model.generate_cellGenesis(
                input_ids=input_ids,
                expression_level=input_vals,
                max_new_tokens=max_new_tokens + input_ids.shape[1],
                top_k=top_k,
                override_gene_sequence=override_gene_sequence if cheat_with_tokens else None,
                override_expr_sequence=override_expr_sequence if cheat_with_expr else None,
                **generate_kwargs
            )
            if verbose:
                print(f"  Model generation took {time.time() - t3:.2f} sec")
    
            # === Post-Processing ===
            t4 = time.time()
            for b in range(len(batch_indices)):
                plen = len(prompt_ids_list[b])
                gen_seq = generated_ids[b].tolist()
                gen_vals = generated_vals[b].tolist()
                gen_vals_binned = generated_vals_binned[b].tolist()
    
                new_ids = gen_seq[plen:]
                new_vals = gen_vals[plen:]
                gene_tokens = self.tokenizer.convert_ids_to_tokens(new_ids)
    
                row = self.tokens_and_vals_to_expression_row(
                    var_names=self.adata.var_names.tolist(),
                    gene_tokens=gene_tokens,
                    new_vals=new_vals,
                    return_series=False
                )
    
                if return_gt:
                    tgt_tokens = torch.tensor(target_tokens_list[b], dtype=torch.long, device=self.device).unsqueeze(0)
                    tgt_vals = torch.tensor(target_vals_list[b], dtype=torch.long, device=self.device).unsqueeze(0)
                    tgt_real_vals = torch.tensor(target_real_vals_list[b], dtype=torch.float32, device=self.device).unsqueeze(0)
                    results.append((row, gene_tokens, new_vals, gen_seq, gen_vals_binned, gen_vals, tgt_tokens, tgt_vals, tgt_real_vals))
                else:
                    results.append((row, gene_tokens, new_vals, gen_seq, gen_vals_binned, gen_vals))
            if verbose:
                print(f"  Post-processing took {time.time() - t4:.2f} sec")
                print(f"Total batch time: {time.time() - batch_start_time:.2f} sec")
    
        return results




    def tokens_and_vals_to_expression_row(
        self,
        var_names: List[str],
        gene_tokens: List[str],
        new_vals: List[float],
        return_series: bool = False
    ) -> Union[np.ndarray, pd.Series]:
        """
        Build an expression‐vector aligned to var_names from
        a list of gene_tokens + real‐valued new_vals.
        
        - Missing genes → 0
        - Duplicates → average over their values
        """
        # 1) Aggregate values per gene
        agg = {}
        for g, v in zip(gene_tokens, new_vals):
            agg.setdefault(g, []).append(v)
        # 2) compute mean for each gene
        avg = {g: sum(vals) / len(vals) for g, vals in agg.items()}
    
        # 3) build the output row
        expr = np.zeros(len(var_names), dtype=float)
        for i, gene in enumerate(var_names):
            expr[i] = avg.get(gene, 0.0)
    
        if return_series:
            return pd.Series(expr, index=var_names)
        return expr

        
    def is_cell_type_entity(self, token_entity):
        return token_entity in self.mulan_cell_type_entities
    
    def cuda_count(self,):
        print(f'scMulan is currently available to {torch.cuda.device_count()} GPUs.')
        return torch.cuda.device_count()

    def check_adata(self, adata, force=False): # set force as True to pass check adata anyway.

        if force:
            print('✅ forcing pass check')
            print("👸 scMulan is ready")
            self.adata = adata.copy()
            return True
        # check normalize and log1p
        adata_max = adata.X.max()
        assert adata_max < 10, f'🚫 Please make sure adata is processed with normalization (sum = 1e4) and log1p, your adata max is {adata_max}.'
        # check gene symbol uniform
        adata_var = set(adata.var_names.tolist())
        mulan_geneset = set(self.meta_info['gene_set'])
        count = len(adata_var.intersection(mulan_geneset))
        assert count == len(self.meta_info['gene_set']), f'🚫 Please make sure adata is processed with uniformed gene symbol, your gene set has {count} overlap with scMulan.'
        # use mulan gene set
        self.adata = adata[:,self.mulan_gene_set].copy()
        print('✅ adata passed check')
        print("👸 scMulan is ready")
        
        

    def process_data_in_parallel(self, func, n_process, save_dir = None):

        # idxs = np.array_split(np.arange(self.adata.n_obs), n_process)
        idxs = np.array_split(self.adata.obs_names, n_process)
        
        devices = [i % torch.cuda.device_count() for i in range(n_process)]
        args = []
        for idx_subset, device_id, proc_id in zip(idxs, devices, range(n_process)):
            if save_dir:
                save_path = os.path.join(save_dir, f"process_{proc_id}.pt")
            else:
                save_path = None
            args.append((idx_subset, device_id, save_path))

        with multiprocessing.Pool(n_process) as pool:
            results = pool.starmap(func, args)
        combined_results = [item for sublist in results for item in sublist]

        return combined_results
    


def model_inference(ckp_path: str,
                    adata: AnnData,
                    meta_info_path: str = os.path.join(os.path.dirname(__file__), 'utils', 'meta_info.pt'),
                    kv_cache: Optional[bool] = False,
                    **kwargs,
                    ):
    
    ckp = torch.load(ckp_path, map_location='cpu')
    gptconf = MulanConfig(**ckp['model_args'])
    if kv_cache:
        model = scMulanModel_kv(gptconf)
    else:
        model = scMulanModel(gptconf)
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(ckp['model'])
    model.eval()
    model.hidden_dim = ckp['model_args']['n_embd']
    # model.half()
    meta_info = torch.load(meta_info_path)
    tokenizer = scMulanTokenizer(meta_info['token_set'])
    n_express_level = ckp['model_args']['expression_level']

    scml = scMulan(model,adata,meta_info,tokenizer,n_express_level,**kwargs)

    return scml

def model_generate(ckp_path: str,
                    adata: AnnData,
                    meta_info: dict,
                    use_kv_cache: bool=False,
                    **kwargs,
                    ):
    
    ckp = torch.load(ckp_path, map_location='cpu')
    ckp['model_args']['expression_level'] = 100
    gptconf = MulanConfig(**ckp['model_args'])
    print(gptconf)
    gptconf.vocab_size = len(meta_info['token_set'])
    bin_edges = compute_global_bin_edges(adata, meta_info['gene_set'],gptconf.expression_level)
    gptconf.bin_edges = bin_edges

    if use_kv_cache == False:
        model = scMulanModel(gptconf)
    else:
        model = scMulanModel_kv(gptconf)
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(ckp['model'])
    model.eval()
    model.hidden_dim = ckp['model_args']['n_embd']
    # model.half()
    
    tokenizer = scMulanTokenizer(meta_info['token_set'])
    n_express_level = ckp['model_args']['expression_level']
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    scml = scMulan(adata,meta_info,tokenizer,n_express_level,bin_edges=bin_edges,model=model,**kwargs)

    return scml


def fine_tuning(
                    adata: AnnData,
                    meta_info: dict,
                    n_express_level: int = 10,
                    **kwargs
                    ):

    tokenizer = scMulanTokenizer(meta_info['token_set'])
    bin_edges = compute_global_bin_edges(adata, meta_info['gene_set'],n_express_level)

    scml = scMulan(adata,meta_info,tokenizer,n_express_level, bin_edges=bin_edges, **kwargs)

    return scml


def generate_prompt_for_cg(
    idx: int,
    meta_data: pd.DataFrame,
    meta_pool: dict,
    tokenizer: scMulanTokenizer,
    add_xyz_noise: bool = False,
    min_max_bounds=None
) -> Tuple[List[int], List[int]]:
    """
    Build prompt IDs by scanning meta_pool for keys that are actual
    columns in meta_data.  For each matching key, if the cell's
    value is in meta_pool[key], emit (col_id, val_id).

    Returns
    -------
    prompt_ids : List[int]
        [ col1_id, col2_id, ... ]
    prompt_vals: List[int]
        [ val1_id, val2_id, ... ]
    """
    prompt_ids: List[int] = []
    prompt_vals: List[int] = []

    col_positions = {col: i for i, col in enumerate(meta_data.columns)}

    for col, allowed_vals in meta_pool.items():
        # only consider columns you actually have
        if col not in meta_data.columns:
            continue
            
        col_idx = col_positions[col]
        val = meta_data.iat[idx, col_idx]
        # skip NA or filler
        if pd.isna(val) or str(val) == 'Unclassified':
            continue

        # if you’ve specified an allow‐list, enforce it
        if allowed_vals is not None and val not in allowed_vals:
            continue

        # convert column name → token ID
        col_id = tokenizer.convert_tokens_to_ids([str(val)])[0]

        prompt_ids.append(col_id)
        prompt_vals.append(0)

    # for coord in ['<x>','<y>','<z>']:
    #     if coord in meta_data.columns:
    #         col_idx = col_positions[coord]
    #         val = meta_data.iat[idx, col_idx]
    #         prompt_ids.append(tokenizer.convert_tokens_to_ids([coord])[0])
    #         prompt_vals.append(val)

    for coord in ['<x>', '<y>', '<z>']:
        if coord in meta_data.columns:
            col_idx = col_positions[coord]
            val = meta_data.iat[idx, col_idx]

            if add_xyz_noise:
                val = int(val)  # ensure it's an integer
                noise = np.random.choice([-1, 0, 1])  # discrete noise
                val += noise
                min_val, max_val = min_max_bounds
                val = max(min_val, min(val, max_val))  # clamp

            prompt_ids.append(tokenizer.convert_tokens_to_ids([coord])[0])
            prompt_vals.append(val)
        

    return prompt_ids, prompt_vals    

def compute_global_bin_edges(
    adata: AnnData,
    mulan_gene_set: List,
    n_express_level: int,
    include_0s: bool = False
) -> np.ndarray:
    """
    Scan your entire expression matrix to find the global min/max of all
    positive (or optionally zero‐included) values, then build
    self.n_express_level bins.

    Stores the result in self.bin_edges of length (n_bins+1).
    """
    adata_var = set(adata.var_names.tolist())
    adata = adata[:,mulan_gene_set]
    cellDFHVG = pd.DataFrame(adata.X.toarray(), columns = mulan_gene_set)
    cellDFHVG.index = list(adata.obs.index)
    adata_matrix = cellDFHVG
    # Flatten all values (optionally including zeros)
    if include_0s:
        vals = adata_matrix.values.ravel()
    else:
        # only positives
        vals = adata_matrix.values.ravel()
        vals = vals[vals > 0]

    if len(vals) == 0:
        raise ValueError("No positive expression values found to build edges")

    # compute uniform edges
    min_val = vals.min()
    max_val = vals.max()
    # +1 so that digitize returns 1..n_bins for positives
    edges = np.linspace(min_val, max_val, n_express_level + 1)
    
    return edges
        

    



    

        