---
license: mit
tags:
  - spatial-transcriptomics
  - single-cell
  - generative-model
  - biology
---

# MIMYR pretrained models

Pretrained checkpoints for [MIMYR](https://github.com/gkrieg/mimyr), a generative
model of missing tissue in spatial transcriptomics. Given neighboring tissue
sections registered to the Allen Common Coordinate Framework, MIMYR generates an
unseen section in three stages: cell locations (a conditional diffusion model), cell
types (a classifier), and gene expression (an autoregressive transformer).

These are the models behind the results in the paper. Each one is a set of three
checkpoints, selected in MIMYR with `--pretrained`:

| `--pretrained` | Paper | Trained on | Use with |
|---|---|---|---|
| `merscope` (default) | Fig. 2 | MERSCOPE whole-brain atlas | `--data_mode rq1` or `demo` |
| `zhuang1` | Fig. 3 (zero-shot) | Zhuang-ABCA-1 | `--data_mode rq3_v2` |
| `zhuang2-finetuned` | Fig. 3 | Zhuang-ABCA-1, fine-tuned on Zhuang-ABCA-2 | `--data_mode rq3_v2` |
| `zhuang3-finetuned` | Fig. 4 | fine-tuned on Zhuang-ABCA-3 (sagittal) | `--data_mode rq4` |

Selecting a model also applies the inference settings used for it in the paper
(reference-guidance strength, `--omit_x`), unless you override them.

## Usage

```bash
git clone https://github.com/gkrieg/mimyr && cd mimyr
pip install -e .
python main.py                                   # demo: merscope model on a small bundled dataset
python main.py --data_mode rq1                   # the full MERSCOPE benchmark
python main.py --pretrained zhuang3-finetuned --data_mode rq4
```

Checkpoints are downloaded from this repository on first use and cached by
`huggingface_hub`. The benchmark sections come from the companion dataset
[gkrieg13/mimyr-data](https://huggingface.co/datasets/gkrieg13/mimyr-data).

## Files

| File | Model | Used by |
|---|---|---|
| `location/smoothtune_conditional_ddpm_2d_checkpoint_400.pt` | location DDPM | `merscope`, `zhuang3-finetuned` |
| `location/location_zhuang1_ddpm_2d_checkpoint_400.pt` | location DDPM | `zhuang1`, `zhuang2-finetuned` |
| `celltype/best_model_rq1.pt` | cell-type classifier | `merscope`, `zhuang3-finetuned` |
| `celltype/celltype_zhuang1.pt` | cell-type classifier | `zhuang1`, `zhuang2-finetuned` |
| `expression/*_epoch*_model.pt` | expression transformer (28M parameters) | one per model |

The expression checkpoints keep their training epoch in the filename because
MIMYR resumes epoch numbering from it when fine-tuning.

## Citation

> Deshpande A, Bei Z, Ma J, Krieger S. MIMYR: Generative modeling of missing tissue
> in spatial transcriptomics. *bioRxiv* [Preprint]. 2025 Nov 27:2025.11.24.690239.
> doi: [10.1101/2025.11.24.690239](https://doi.org/10.1101/2025.11.24.690239).
> PMID: 41394599; PMCID: PMC12697367.
