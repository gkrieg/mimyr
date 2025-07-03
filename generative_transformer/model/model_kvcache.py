from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.generation.utils import SampleDecoderOnlyOutput


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, layer_past=None):
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        B, T, C = k.size()
        _, Tq, _ = q.size()

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        if Tq < k.size(2):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None):
        a, kv = self.attn(self.ln_1(x), layer_past)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, kv


@dataclass
class MulanConfig:
    block_size: int = 1000
    vocab_size: int = 1011
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    train_mode: str = 'pretrain'
    expression_level: int = 10
    ele: int = 1
    bin_edges: torch.Tensor = None


class scMulanModel_kv(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.expression_level is not None
        assert config.ele == 1
        self.config = config
        if config.bin_edges is not None:
            self.bin_edges = torch.tensor(config.bin_edges)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wee = nn.Embedding(config.expression_level + 1, config.n_embd), # +1 for non gene tokens
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.epx_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # expr level

        self.epx_regressor = nn.Sequential(
            nn.Linear(config.vocab_size, config.n_embd),  # map vocab→hidden
            nn.ReLU(),
            nn.Linear(config.n_embd, 1)                   # hidden→scalar
        )

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        # super().__init__()
        # assert config.ele == 1
        # self.config = config
        # if config.bin_edges is not None:
        #     self.bin_edges = config.bin_edges

        # self.transformer = nn.ModuleDict(dict(
        #     wte=nn.Embedding(config.vocab_size, config.n_embd),
        #     wee=nn.Embedding(config.expression_level + 1, config.n_embd),
        #     drop=nn.Dropout(config.dropout),
        #     h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        #     ln_f=LayerNorm(config.n_embd, bias=config.bias)
        # ))

        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.epx_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.epx_regressor = nn.Sequential(
        #     nn.Linear(config.vocab_size, config.n_embd),
        #     nn.ReLU(),
        #     nn.Linear(config.n_embd, 1)
        # )

        # if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
        #     logger.info(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def to(self, device):
        model = super().to(device)
        if hasattr(model, "bin_edges"):
            model.bin_edges = model.bin_edges.to(device)
        return model

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Grow (or shrink) both the input token embeddings and the LM head
        to `new_num_tokens`, copying over old weights and randomly
        initializing any new rows.
        """
        old_emb = self.transformer.wte.weight.data
        old_num, emb_dim = old_emb.shape

        # 1) New embedding layer
        new_emb = nn.Embedding(new_num_tokens, emb_dim)
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(old_emb)
            # rows [old_num:] are randomly init per nn.Embedding default
        self.transformer.wte = new_emb

        # 2) New LM head (tied weights in many HF models)
        old_head = self.lm_head.weight.data  # shape (old_num, emb_dim)
        new_head = nn.Linear(emb_dim, new_num_tokens, bias=False)
        with torch.no_grad():
            new_head.weight[:old_num].copy_(old_head)
        self.lm_head = new_head

        old_head2 = self.epx_head.weight.data  # shape (old_num, emb_dim)
        new_head2 = nn.Linear(emb_dim, new_num_tokens, bias=False)
        with torch.no_grad():
            new_head2.weight[:old_num].copy_(old_head2)
        self.epx_head = new_head2

        # 3) Update config
        self.config.vocab_size = new_num_tokens

        self.epx_regressor = nn.Sequential(
                nn.Linear(self.config.vocab_size, self.config.n_embd),  # map vocab→hidden
                nn.ReLU(),
                nn.Linear(self.config.n_embd, 1)                   # hidden→scalar
            )

    def resize_expression_embeddings(self, new_expression_level: int):
        """
        Grow (or shrink) the expression‐level embedding `wee`
        to support values 0…new_expression_level (inclusive),
        copying old weights and randomly initializing any new rows.
        """
        # how many bins we used to have (including zero)
        old_emb = self.transformer.wee.weight.data
        old_num, emb_dim = old_emb.shape

        # how many we now want (remember wee was built with +1)
        new_num = new_expression_level + 1

        # 1) new embedding
        new_emb = nn.Embedding(new_num, emb_dim)
        with torch.no_grad():
            # copy over the old rows
            new_emb.weight[:old_num].copy_(old_emb)
            # rows [old_num:] are freshly initialized
        self.transformer.wee = new_emb

        # 2) update config so that later calls (or re-initializations) know the new max
        self.config.expression_level = new_expression_level

    def forward(self, idx=None, inputs_embeds=None, past_key_values=None, x_expr=None, return_hidden=False):
        if past_key_values is None:
            past_key_values = [None] * len(self.transformer.h)

        if idx is not None:
            tok_emb = self.transformer.wte(idx)
        else:
            tok_emb = inputs_embeds

        expr_emb = self.transformer.wee(x_expr)
        x = self.transformer.drop(tok_emb + expr_emb)

        presents_kv = []
        for block, past in zip(self.transformer.h, past_key_values):
            x, kv = block(x, past)
            presents_kv.append(kv)

        x = self.transformer.ln_f(x)

        logits_labels = self.lm_head(x)
        logits_exp = self.epx_head(x)

        B, T, V = logits_exp.size()
        flat_exp = logits_exp.view(-1, V)             # (B*T, V)
        flat_reg = self.epx_regressor(flat_exp)       # (B*T, 1)
        logits_reg = flat_reg.view(B, T, 1)           # (B, T, 1)

        return logits_labels, logits_reg, tuple(presents_kv)

    @torch.no_grad()
    def generate_cellGenesis(self, input_ids, expression_level, max_new_tokens, ignore_Idx=None, top_k=None, gamma=1.0, override_gene_sequence=None, override_expr_sequence=None, verbose=False):
        output_idx = input_ids
        output_expr = expression_level
        real_expr = expression_level.float()
        past_key_values = None

        predicted_gene_tokens = []
        predicted_expression_bins = []
        predicted_real_expr = []

        while output_idx.shape[1] < max_new_tokens:
            logits_cls, logits_exp, past_key_values = self(idx=output_idx[:, -1:], x_expr=output_expr[:, -1:], past_key_values=past_key_values)
            
            logits_cls = logits_cls[:, -1, :]
            logits_exp = logits_exp[:, -1].float() # (B,)

            if ignore_Idx is not None:
                logits_cls[:, ignore_Idx] = float('-inf')
            seen = output_idx[:, :-1]
            for b in range(seen.size(0)):
                logits_cls[b, seen[b]] = float('-inf')

            if top_k is not None:
                v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)), dim=-1)
                logits_cls[logits_cls < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits_cls, dim=-1)
            probs[:, 0] *= gamma
            next_token = torch.multinomial(probs, num_samples=1)

            next_expr_real = logits_exp
            bin_next = torch.bucketize(next_expr_real, self.bin_edges)
            bin_next = torch.clamp(bin_next, 0, self.bin_edges.numel() - 1)

            predicted_gene_tokens.append(next_token)
            predicted_expression_bins.append(bin_next)
            predicted_real_expr.append(next_expr_real)

            step_idx = output_idx.size(1)

            if override_gene_sequence is not None and step_idx < override_gene_sequence.size(1):
                next_token_input = override_gene_sequence[:, step_idx].unsqueeze(1)
            else:
                next_token_input = next_token

            if override_expr_sequence is not None and step_idx < override_expr_sequence.size(1):
                next_expr_input = override_expr_sequence[:, step_idx].unsqueeze(1)
            else:
                next_expr_input = bin_next

            output_idx = torch.cat([output_idx, next_token_input], dim=1)
            output_expr = torch.cat([output_expr, next_expr_input], dim=1)
            real_expr = torch.cat([real_expr, next_expr_real], dim=1)

            if (next_token == 0).all():
                break

        predicted_gene_tokens = torch.cat(predicted_gene_tokens, dim=1)
        predicted_expression_bins = torch.cat(predicted_expression_bins, dim=1)
        predicted_real_expr = torch.cat(predicted_real_expr, dim=1)

        return predicted_gene_tokens, predicted_expression_bins, predicted_real_expr


@dataclass
class SampleDecoderOutput(SampleDecoderOnlyOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    expression: Optional[torch.LongTensor] = None
