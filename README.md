![MIMYR Method](mimyr.jpg)

# MIMYR

MIMYR is a generative framework for spatial transcriptomics. Given neighboring
tissue sections registered to the Allen Common Coordinate Framework (CCF), it
generates an unseen section: where the cells are, what type each cell is, and
what genes each cell expresses.

It has three components, run in sequence:

1. **Location model**: a conditional diffusion model (DDPM) that samples cell
   coordinates in the plane of the new section, with optional backward guidance
   from a KDE fit to a neighboring reference section.
2. **Cell-type model**: a classifier that predicts each cell's type from its CCF
   coordinates.
3. **Expression model**: an autoregressive transformer that generates each cell's
   expression profile, conditioned on its location and cell type.

---

## Installation

MIMYR requires Python 3.10 or newer. It is not on PyPI yet, so install it from
source.

```bash
conda create --name mimyr python=3.10
conda activate mimyr
```

If you have an NVIDIA GPU, first install the PyTorch build that matches your CUDA
version (see [pytorch.org](https://pytorch.org/get-started/locally/)); otherwise
`pip` installs the default build.

Then, from the repository root:

```bash
pip install -e .
```

This installs the `mimyr` package and its dependencies. The gene and cell-type
metadata the models need ship with the package.

---

## Data and pretrained checkpoints

On first run, `main.py` downloads what it needs from Google Drive if it is missing:

| Download | Size | When | Contents |
|---|---|---|---|
| `data.zip` | 4.1 GB | the `--data_dir` path does not exist | `data/subclass_z1_d338_0_rotated/` (rq1 sections), `data/Zhuang-ABCA-1/` … `Zhuang-ABCA-4/`, the matching `ccf-ABCA-*.csv` files, `data/cleaned_versions/` |
| `model_checkpoints.zip` | 830 MB | `./model_checkpoints` does not exist | the default location, cell-type, and expression checkpoints, plus cell-type (`best_model_rq{1,3,4,5}.pt`) and expression (`Zhuang-2`, `Zhuang-3`, `Trem2_5xFAD`) checkpoints for the other experiments |

Both archives are extracted into the **current working directory**, whatever
`--data_dir` is set to, so run `main.py` from the directory where you want `data/`
and `model_checkpoints/` to live.

### Input format

Each section is an `.h5ad` file (`.slaf` is also supported) with:

- `obsm["aligned_spatial"]`: CCF coordinates
- `obs["cluster"]`: cell-type labels (choose a different column with `--data_label`).
  Labels are mapped to tokens using the bundled Allen Brain Cell atlas vocabulary,
  so they must use ABC-atlas names.
- `X`: the gene expression matrix

---

## Running inference

`main.py` is the entry point. It loads a dataset according to `--data_mode`,
generates each test section from its neighboring reference sections, scores the
result against the real section, and writes the metrics to a CSV.

Each of the three stages is controlled by its own flag:

| Flag | Default | Values |
|---|---|---|
| `--location_inference_type` | `skip` | `model` (DDPM), `closest_slice` (copy the nearest reference section) |
| `--cluster_inference_type` | `model` | `model` (classifier), `majority_baseline` |
| `--expression_inference_type` | `end` | `model` (transformer), `lookup` (nearest same-type reference cell), `averaging` |

Every stage also accepts two special values:

- `skip` uses the **ground truth** for that stage instead of predicting it
  (useful for evaluating one stage in isolation).
- `end` stops the pipeline after the previous stage.

With the defaults, `python main.py` runs the rq1 experiment using ground-truth
locations, predicted cell types, and no expression generation. To run the full
pipeline:

```bash
python main.py \
  --data_mode rq1 \
  --location_inference_type model \
  --cluster_inference_type model \
  --expression_inference_type model \
  --metrics soft_spearman_correlation,soft_f1 \
  --metric_sampling 100 \
  --out_csv results/rq1.csv
```

The baselines used in the paper:

```bash
python main.py --data_mode rq1 \
  --location_inference_type closest_slice \
  --cluster_inference_type majority_baseline \
  --expression_inference_type lookup \
  --metrics soft_spearman_correlation,soft_f1 --metric_sampling 100
```

`main_experimental_results.sh` lists the commands behind the paper's experiments.
Several of them point at checkpoints on the lab's cluster that are not in the
download.

### Data modes

`--data_mode` selects the dataset and the train/validation/test/reference split:

| Mode | Dataset | Paths it reads |
|---|---|---|
| `rq1` | coronal sections from one brain, fixed held-out sections | `--data_dir/subclass_z1_d338_0_rotated/` |
| `rq3`, `rq3_v2`, `rq3_v2_d1` … `rq3_v2_d4` | Zhuang-ABCA-2 (coronal); the `_d*` modes vary the distance to the reference sections | `--zhuang_data_dir/Zhuang-ABCA-2/`, `--data_dir/ccf-ABCA-2.csv` |
| `rq4`, `rq4_rq3`, `rq4_noref` | Zhuang-ABCA-3 (sagittal); `rq4_noref` uses all of Zhuang-ABCA-1 as the reference pool | `--zhuang_data_dir/Zhuang-ABCA-3/` (and `-1`, `-2`), `--data_dir/ccf-ABCA-*.csv` |
| `rq5` | diseased (Trem2 / 5xFAD) sections | `--diseased_data_dir` |
| `base_zhuang1` | Zhuang-ABCA-1, for training the base models | `--zhuang_data_dir/Zhuang-ABCA-1/` |

With the downloaded data, pass `--zhuang_data_dir data` for the Zhuang modes.

### Main options

- `--location_model_checkpoint`, `--cluster_model_checkpoint`, `--expression_model_checkpoint`:
  checkpoint paths (defaults: the files in `model_checkpoints/`)
- `--guidance_signal`: strength of KDE backward guidance for the location model
  (default `0.0`, off); `--kde_bandwidth` sets the KDE bandwidth (default `0.01`)
- `--metrics`: comma-separated metrics (default `soft_accuracy`). Available:
  `soft_accuracy`, `soft_spearman_correlation`, `soft_correlation_top_100`, `soft_f1`,
  `gridized_l1_distance`, `gridized_kl_divergence`, `celltype_abundance`,
  `composition_entropy`, `composition_pairwise`, `delauney_colocalization`,
  `neighborhood_enrichment`
- `--metric_sampling`: percentage of ground-truth cells the metrics are computed
  on (default `1`; the paper uses `100`)
- `--metric_gene_set_file`: restrict expression metrics to the genes listed in a
  file, one per line (e.g. `gene_overlap.txt`); `--metric_filter_by_gt` restricts
  them to genes expressed in the ground truth
- `--eval_workers`: number of sections scored in parallel (default `1`)
- `--device`: `cuda` (default) or `cpu`

Expression decoding can be tuned with `--top_k`,
`--expression_select_max_index_margin`, and the weak teacher forcing options
(`--expression_weak_teacher_forcing`, `--wtf_*`), which pull generation toward the
nearest same-type reference cell. All of these are off or neutral by default; see
`python main.py --help` for the full list.

### Output

- `--out_csv` (default `results/output.csv`): one row per test section, containing
  the full configuration and the metrics. Rows are appended, and a section is
  **skipped** if a row with the same configuration is already in the CSV.
- `--artifact_dir` (default `artifacts/`): one timestamped folder per test section
  with `config.json`, `results.json`, `pred.pkl` (the generated AnnData), and plots
  of the predicted and ground-truth locations and cell types.

---

## Training

### Expression model

`--run_mode train` trains the expression transformer with distributed
data-parallel training. Launch it with `torchrun`:

```bash
torchrun --nproc_per_node=<NUM_GPUS> main.py \
  --run_mode train \
  --skip_combined_fit \
  --data_mode rq1 \
  --data_dir /path/to/data \
  --expression_model_checkpoint model_checkpoints/TG-base4_epoch4_model.pt \
  --expression_from_finetuned \
  --expression_output_dir /path/to/save/checkpoints \
  --expression_epochs 100 \
  --expression_batch_size 128
```

- Training and validation data come from the `--data_mode` split, the same as for
  inference. The preprocessed data is cached in `--expression_output_dir`, so later
  runs with the same output directory skip preprocessing.
- A checkpoint is saved every `--expression_save_frequency` epochs (default `10`).
- `--expression_from_finetuned` is what loads the checkpoint's **weights**; it also
  resumes the epoch count from the `epochN` in the checkpoint's filename. Without
  it, only the checkpoint's architecture is used and training starts from random
  weights. Pass it whenever you want to continue from a checkpoint.
- To train from scratch without a checkpoint, pass `--expression_model_checkpoint ""`
  together with `--expression_model_size small|medium|large`.
- Training logs to Weights & Biases. Set `WANDB_MODE=offline` to log locally.

Other useful options: `--expression_lr` (default `5e-5`), `--expression_lambda_val`
(weight of the regression loss, default `1.0`), `--expression_max_len` (default
`512`), `--expression_epoch_samples` (cells per epoch; `-1`, the default, uses all of
them), `--expression_xyz_noise` / `--expression_xyz_noise_magnitude` (coordinate
noise), and `--expression_adata2` (an extra `.h5ad`, e.g. scRNA-seq, merged into
training).

### Cell-type and location models

Without `--skip_combined_fit`, `--run_mode train` first trains the cell-type
classifier on the `.h5ad`/`.slaf` files in `--training_slice_directory` and
`--val_slice_directory` (default `data/cleaned_versions`). These files also need an
`obs["token"]` column. The best model is saved to `./best_model.pt`.

`main.py` does not train the location model. `train_location_zhuang1.py` and
`train_celltype_zhuang1.py` retrain the location and cell-type models on
Zhuang-ABCA-1; they currently contain cluster-specific data paths.

### Configuration files

Any option can be put in a YAML file and passed with `--config`. Each key is an
argument name (for example `data_dir:` or `expression_model_checkpoint:`), and flags
given on the command line override the file:

```bash
python main.py --config config.yaml
torchrun --nproc_per_node=4 main.py --config configs/train_base.yaml --run_mode train
```

`config.yaml` and the files in `configs/` hold the settings for the paper's
experiments. They contain absolute paths on the lab's cluster, so copy one and
change the paths before using it.

---

## Project structure

```
MIMYR/
├── pyproject.toml               # Package definition (pip install -e .)
├── main.py                      # Entry point: inference and training
├── config.yaml                  # Example configuration
├── configs/                     # Per-experiment configurations
├── main_experimental_results.sh # Commands for the paper's experiments
├── gene_overlap.txt             # Shared gene set for rq1/rq3 evaluation
├── train_location_zhuang1.py    # Retrain the location model on Zhuang-ABCA-1
├── train_celltype_zhuang1.py    # Retrain the cell-type model on Zhuang-ABCA-1
└── src/mimyr/                   # The installable `mimyr` package
    ├── data_loader.py           # SliceDataLoader: data modes, splits, preprocessing
    ├── inference.py             # The three-stage generation pipeline
    ├── evaluator.py             # Runs the selected metrics
    ├── metrics.py               # Metric implementations
    ├── gene_exp_model.py        # Cell-type label to token mapping
    └── models/
        ├── diffusion_model.py        # Location model (DDPM)
        ├── celltype_model.py         # Cell-type classifier
        ├── biological_model.py       # KDE used for location guidance
        ├── combined_model.py         # Loads the location and cell-type models
        └── generative_transformer/   # Expression model
            ├── Mimyr.py              # Generation and inference helpers
            ├── finetune_mimyr.py     # Training loop (DDP)
            ├── data_util.py          # Data harmonization and loaders
            ├── metadata/             # Bundled gene and cell-type metadata
            ├── model/                # Transformer implementation
            ├── reference/            # Gene symbol reference list
            └── utils/                # Tokenizer
```

## License

MIMYR is released under the [MIT License](LICENSE).
