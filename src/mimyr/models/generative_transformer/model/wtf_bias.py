"""Weak teacher forcing (WTF) logit biases for inference-time generation.

These helpers softly steer autoregressive cell generation toward a per-cell
"prior" expression vector copied from the nearest same-cell-type reference cell
(the lookup baseline). The biases are pre-softmax logit additions, so they only
re-rank tokens the model already considers legal — adding a finite bias to a
``-inf`` (masked) logit leaves it masked. With all alphas/sigma at 0 (or the
prior tensors at ``None``) these functions are exact no-ops, so generation is
byte-identical to standard free-running inference.

Conventions
-----------
All tensors are already sliced to the current model-call batch rows:
  * ``present_mask_rows`` : (R, V) bool   -- True at gene-token ids the prior
                                              wants expressed.
  * ``prior_bin_rows``    : (R, V) long   -- lookup expression bin at each gene
                                              token id (0 elsewhere / unexpressed).
  * ``prior_valid_rows``  : (R,)   bool   -- False for cells with no reference
                                              neighbor (those rows get zero bias).
``R`` is the number of rows passed in (full batch for the non-cache / non-fast
loops; the active subset for the pruned fast loop).
"""

import torch


def apply_gene_presence_bias(
    logits_cls_step,
    present_mask_rows,
    prior_valid_rows,
    alpha_gene,
    eos_id,
):
    """Add ``alpha_gene`` to gene-token logits the lookup prior expresses.

    Must be called AFTER the seen-token and strictly-decreasing-order masks
    (so masked ``-inf`` logits stay ``-inf``) and BEFORE the softmax. The EOS
    column is never biased, so termination length is not perturbed.
    """
    if present_mask_rows is None or alpha_gene == 0.0:
        return logits_cls_step
    bias = present_mask_rows.to(logits_cls_step.dtype) * float(alpha_gene)
    bias[:, eos_id] = 0.0
    bias = bias * prior_valid_rows.to(logits_cls_step.dtype).unsqueeze(1)
    return logits_cls_step + bias


def apply_bin_value_bias(
    logits_bins_step,
    next_token,
    prior_bin_rows,
    prior_valid_rows,
    alpha_bin,
    bin_sigma,
    unexpressed_mode,
):
    """Bias the per-step expression-bin logits toward the prior's bin.

    Must be called AFTER ``next_token`` is finalized (the chosen gene token
    selects which prior bin to steer toward) and BEFORE the bin-decode block,
    so it applies under every ``MIMYR_BIN_DECODE`` mode.

    ``bin_sigma > 0`` adds an ordinal-aware Gaussian bonus over bin indices
    centered at the prior bin; otherwise ``alpha_bin`` is added to the single
    prior-bin logit. ``unexpressed_mode``:
      * ``"none"`` -- bias only where the prior expresses the gene (bin > 0).
      * ``"low"``  -- also bias toward bin 0 where the prior is unexpressed.
    """
    if prior_bin_rows is None or (alpha_bin == 0.0 and bin_sigma == 0.0):
        return logits_bins_step

    E = logits_bins_step.size(-1)
    # Chosen gene's prior bin per row; eos / unseen tokens index a 0 column.
    bstar = prior_bin_rows.gather(1, next_token.clamp_min(0))  # (R, 1) long
    bstar = bstar.clamp(0, E - 1)

    if unexpressed_mode == "low":
        apply = prior_valid_rows.unsqueeze(1)  # (R, 1) -- bstar==0 -> bias bin 0
    else:
        apply = prior_valid_rows.unsqueeze(1) & (bstar > 0)

    if bin_sigma > 0.0:
        levels = torch.arange(
            E, device=logits_bins_step.device, dtype=logits_bins_step.dtype
        ).unsqueeze(0)  # (1, E)
        gauss = torch.exp(
            -0.5 * ((levels - bstar.to(logits_bins_step.dtype)) / float(bin_sigma)) ** 2
        )
        bias = float(alpha_bin) * gauss  # (R, E)
    else:
        bias = torch.zeros_like(logits_bins_step)
        bias.scatter_(1, bstar, float(alpha_bin))

    bias = bias * apply.to(logits_bins_step.dtype)
    return logits_bins_step + bias


# ---------------------------------------------------------------------------
# Probability-space guidance (tunable 0 = ignore reference .. 1 = exact override)
# ---------------------------------------------------------------------------
# Unlike the additive-logit biases above (which saturate and can never override
# the model), these mix the model's distribution with a hard reference target in
# PROBABILITY space:  ``p = (1 - s) * p_model + s * p_ref``. With ``s == 0`` (or
# the prior tensors ``None``) they are exact no-ops; with ``s == 1`` the chosen
# gene / bin / value equals the reference cell's, so the generated cell
# reproduces its nearest same-type reference exactly. ``s`` is the per-component
# strength in [0, 1] (the CLI 0..100 knob divided by 100). Cells with
# ``prior_valid == False`` (no reference neighbor) are left untouched.


def mix_gene_probs(probs, present_mask_rows, prior_valid_rows, finite_mask, s_gene, eos_id):
    """Mix the next-gene distribution toward the reference's still-legal genes.

    ``p_ref`` is one-hot on the **largest** token id the reference expresses that
    is still legal at this step (``present_mask & finite_mask``). Choosing the
    max id matters: gene tokens are emitted in strictly decreasing order, so
    targeting the largest remaining reference gene each step emits the whole
    reference set without skipping. When no reference gene remains legal, ``p_ref``
    is one-hot on EOS, so at ``s_gene == 1`` the cell stops with exactly the
    reference gene set. Call on the post-mask softmax ``probs`` (before sampling).
    """
    if present_mask_rows is None or s_gene <= 0.0:
        return probs
    R, V = probs.shape
    ref_support = present_mask_rows & finite_mask  # (R, V) legal reference genes
    ids = torch.arange(V, device=probs.device).unsqueeze(0).expand(R, V)
    masked_ids = torch.where(ref_support, ids, torch.full_like(ids, -1))
    max_id = masked_ids.max(dim=1).values  # (R,), -1 if no legal reference gene
    has_support = max_id >= 0
    target = torch.where(has_support, max_id, torch.full_like(max_id, eos_id))  # (R,)
    p_ref = torch.zeros_like(probs)
    p_ref.scatter_(1, target.unsqueeze(1), 1.0)
    s_row = (float(s_gene) * prior_valid_rows.to(probs.dtype)).unsqueeze(1)  # (R, 1)

    # The model's post-mask distribution can be degenerate (all class logits
    # -inf -> softmax NaN) when top-k keeps only tokens that the strictly-
    # decreasing-order mask then kills. Such a row must lean entirely on the
    # reference, but ``0 * NaN = NaN`` would poison the blend even at s=1. So
    # sanitize the model term, then renormalize the rows we actually steer
    # (s_row > 0). Untouched rows (prior_valid == False) keep their EXACT
    # original ``probs`` so the no-op guarantee holds bit-for-bit.
    probs_safe = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    mixed = (1.0 - s_row) * probs_safe + s_row * p_ref
    row_sum = mixed.sum(dim=1, keepdim=True)
    mixed = mixed / torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    steer = s_row > 0.0  # (R, 1)
    return torch.where(steer, mixed, probs)


def mix_bin_probs(bin_probs, next_token, prior_bin_rows, prior_valid_rows, s_bin, unexpressed_mode):
    """Mix the per-step expression-bin distribution toward the prior's bin.

    ``p_ref`` is one-hot on the reference bin of the chosen gene token. Gating
    mirrors ``apply_bin_value_bias``: ``"none"`` steers only expressed genes
    (bin > 0); ``"low"`` also steers unexpressed genes toward bin 0. Call on the
    bin softmax ``bin_probs``; decode (argmax/expected/sample) from the result.
    """
    if prior_bin_rows is None or s_bin <= 0.0:
        return bin_probs
    E = bin_probs.size(-1)
    bstar = prior_bin_rows.gather(1, next_token.clamp_min(0)).clamp(0, E - 1)  # (R, 1)
    if unexpressed_mode == "low":
        apply = prior_valid_rows.unsqueeze(1)
    else:
        apply = prior_valid_rows.unsqueeze(1) & (bstar > 0)
    p_ref = torch.zeros_like(bin_probs)
    p_ref.scatter_(1, bstar, 1.0)
    s_row = float(s_bin) * apply.to(bin_probs.dtype)  # (R, 1)
    return (1.0 - s_row) * bin_probs + s_row * p_ref


def blend_real_value(v_model, next_token, prior_real_rows, prior_valid_rows, s_value):
    """Blend the regressor's real value toward the reference's real expression.

    ``v_ref`` is the reference cell's (normalized + log1p) expression for the
    chosen gene token; genes the reference does not express contribute 0. This is
    the only path that moves the final magnitude (mean/variance of log
    expression, library size), since the written value is the regressor output,
    not the bin. ``v_model`` and ``next_token`` are both ``(R, 1)``.
    """
    if prior_real_rows is None or s_value <= 0.0:
        return v_model
    v_ref = prior_real_rows.gather(1, next_token.clamp_min(0))  # (R, 1)
    s_row = float(s_value) * prior_valid_rows.to(v_model.dtype).unsqueeze(1)  # (R, 1)
    return (1.0 - s_row) * v_model + s_row * v_ref
