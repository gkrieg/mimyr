# Shared setup for the reproducibility scripts; each script sources this file.
#
# Environment variables (all optional):
#   OUT           where CSVs and generated sections go   (default: results/reproduce)
#   DATA_DIR      where the benchmark data is downloaded (default: data)
#   EVAL_WORKERS  test sections scored in parallel       (default: 4)
#   PYTHON        interpreter that has mimyr installed   (default: python)
#   DRY_RUN       if set, print the main.py commands instead of running them
#
# Arguments given to a script are appended to every main.py call it makes, so
# `bash reproducibility/fig2_rq1.sh --device cpu` runs the whole figure on CPU.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT="${OUT:-results/reproduce}"
DATA_DIR="${DATA_DIR:-data}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
PYTHON="${PYTHON:-python}"
EXTRA=("$@")

mkdir -p "$OUT"

# Metrics behind every soft-Spearman / soft-F1 panel in the paper.
EXPR_METRICS="soft_spearman_correlation,soft_f1"

# The rule-based baseline: copy the nearest reference section's cell positions,
# assign each cell the majority type, and copy the expression of the nearest
# same-type reference cell. Reference guidance only acts on the expression
# model, so it is switched off for the baseline.
BASELINE=(
  --location_inference_type closest_slice
  --cluster_inference_type majority_baseline
  --expression_inference_type lookup
  --no-expression_weak_teacher_forcing
  --wtf_strength_gene 0 --wtf_strength_bin 0 --wtf_strength_value 0
)

FULL_MODEL=(
  --location_inference_type model
  --cluster_inference_type model
  --expression_inference_type model
)

# Reference-guidance strength s on all three channels (gene, bin, value).
guidance() {
  echo --expression_weak_teacher_forcing \
    --wtf_strength_gene "$1" --wtf_strength_bin "$1" --wtf_strength_value "$1"
}

# Local path to a published checkpoint, downloaded from the Hub on first use.
#   hub_ckpt merscope location
hub_ckpt() {
  if [ -n "${DRY_RUN:-}" ]; then echo "<$1 $2 checkpoint>"; return; fi
  "$PYTHON" -c 'import sys; from mimyr import pretrained; print(pretrained.checkpoint(*sys.argv[1:]))' "$1" "$2"
}

# One main.py call that appends a row per test section to $OUT/<csv>.
#   run fig2_rq1.csv --pretrained merscope --data_mode rq1 ...
# main.py skips any section whose configuration is already in the CSV, so an
# interrupted script picks up where it stopped when run again.
run() {
  local csv="$1"
  shift
  local cmd=("$PYTHON" main.py
    --data_dir "$DATA_DIR"
    --metric_sampling 100
    --eval_workers "$EVAL_WORKERS"
    --artifact_dir "$OUT/artifacts"
    --out_csv "$OUT/$csv"
    "$@" ${EXTRA[@]+"${EXTRA[@]}"})
  echo "+ ${cmd[*]}"
  [ -n "${DRY_RUN:-}" ] || "${cmd[@]}"
}
