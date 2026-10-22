# Reproducing the paper's results

The scripts in this folder regenerate the quantitative results of the MIMYR paper
(Deshpande, Bei, Ma and Krieger, *bioRxiv* 2025, doi:10.1101/2025.11.24.690239)
from the published checkpoints and data on the Hugging Face Hub
([gkrieg13/mimyr](https://huggingface.co/gkrieg13/mimyr),
[gkrieg13/mimyr-data](https://huggingface.co/datasets/gkrieg13/mimyr-data)).

## Usage

Install MIMYR as described in the main [README](../README.md#installation), then run
a script from the repository root:

```bash
bash reproducibility/fig2_rq1.sh
```

Checkpoints and data are downloaded on first use. Each script writes one CSV row per
held-out section to `results/reproduce/`, and the generated sections to
`results/reproduce/artifacts/`. The scripts need a GPU; most finish in a few hours on
one NVIDIA A6000.

| Variable | Default | Meaning |
|---|---|---|
| `OUT` | `results/reproduce` | output directory |
| `DATA_DIR` | `data` | where the benchmark data is downloaded |
| `EVAL_WORKERS` | `4` | sections scored in parallel |
| `PYTHON` | `python` | interpreter with `mimyr` installed |
| `DRY_RUN` | unset | print the `main.py` commands without running them |

Arguments given to a script are passed on to every `main.py` call it makes. A script
that stops partway resumes where it left off when run again with the same `OUT`.

## Scripts

| Result | Script | Output |
|---|---|---|
| Fig. 2: MERSCOPE held-out sections | `fig2_rq1.sh` | `fig2_rq1.csv` |
| Fig. 3: Zhuang-ABCA-2, zero-shot and fine-tuned | `fig3_rq3v2.sh` | `fig3_rq3v2.csv` |
| Fig. 3E: distance to the reference sections | `fig3e_distance.sh` | `fig3e_distance.csv` |
| Fig. 4: sagittal Zhuang-ABCA-3 | `fig4_rq4.sh` | `fig4_rq4.csv` |
| Fig. 4E: no-reference sagittal reconstruction | `fig4e_rq4_noref.sh` | `fig4e_noref_{armA,armB,lookup}.csv` |
| Supp.: reference-guidance strength | `supp_ref_guidance.sh` | `supp_ref_guidance_{rq1,rq3}.csv` |
| Supp.: location guidance strength and bandwidth | `supp_location_guidance.sh` | `supp_location_guidance.csv` |
| Supp.: expression decoding | `supp_decoding.sh` | `supp_decoding.csv` |
| Supp.: cell-type sampling temperature | `supp_celltype_temperature.sh` | `supp_celltype_{argmax,sampling}.csv` |
| Supp.: error propagation across stages | `supp_cascade.sh` | `supp_cascade.csv` |
| Supp.: each stage on its own | `supp_stagewise.sh` | `supp_stage_{location,celltype,celltype_class,expression}.csv` |

Expression is scored by soft Spearman and soft F1 at radii 0.03 to 0.15, in columns
such as `soft_spearman_correlation_radius@0.03`; location by binned L1 and KL; cell
types by soft accuracy.

Three analysis scripts work on these outputs and need no GPU:

- `analysis/per_cell_ks.py --results $OUT`: per-cell KS for the reference-guidance,
  decoding, temperature and cascade results (`supp_per_cell_ks.csv`).
- `analysis/celltype_class_accuracy.py`: class-level cell-type accuracy for the
  stage-wise comparison; `supp_stagewise.sh` runs it.
- `analysis/paired_stats.py --results $OUT`: the paired contrasts and t-tests
  reported in the supplement.

SpatialZ, the comparison method in Figs. 2-4, is a separate tool and is not run by
these scripts. `analysis/paired_stats.py --spatialz <csv>` includes it given its
scores.

## Training configurations

`training_configs/` holds the training configuration for each published expression
checkpoint. They record the lab's cluster paths.

| Model | Component | Checkpoint | Configuration |
|---|---|---|---|
| `merscope` | expression | `expression/merscope_epoch165_model.pt` | `merscope_1_base.yaml`, then `merscope_2_finetune.yaml` |
| `zhuang1` | expression | `expression/zhuang1_epoch500_model.pt` | `zhuang1_base.yaml` |
| `zhuang2-finetuned` | expression | `expression/zhuang2-finetuned_epoch502_model.pt` | `zhuang2_finetune.yaml` |
| `zhuang3-finetuned` | expression | `expression/zhuang3-finetuned_epoch510_model.pt` | `zhuang3_finetune.yaml` |
| `zhuang1`, `zhuang2-finetuned` | location | `location/location_zhuang1_ddpm_2d_checkpoint_400.pt` | `python train_location_zhuang1.py --epochs 400 --batch_size 4096` |
| `zhuang1`, `zhuang2-finetuned` | cell type | `celltype/celltype_zhuang1.pt` | `python train_celltype_zhuang1.py --epochs 1000 --batch_size 1024` |
