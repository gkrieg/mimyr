#!/usr/bin/env bash
# Supplementary Fig. (ablations) A: error propagation across stages on
# Zhuang-ABCA-2 (rq3_v2), zero-shot model. Each row replaces one or two stages by
# the ground truth (`skip`) or one stage by its rule-based alternative.
#
# Output: $OUT/supp_cascade.csv, 40 rows (8 configurations x 5 test sections).
# The per-cell KS values in the panel come from analysis/per_cell_ks.py.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CASCADE=(--pretrained zhuang1 --data_mode rq3_v2 --metrics "$EXPR_METRICS")
NO_GUIDANCE=(--no-expression_weak_teacher_forcing
  --wtf_strength_gene 0 --wtf_strength_bin 0 --wtf_strength_value 0)

# location, cell type, expression
for stages in "skip skip model" "model skip model" "skip model model" \
              "model model model" "closest_slice model model" \
              "model majority_baseline model"; do
  set -- $stages
  run supp_cascade.csv "${CASCADE[@]}" --location_inference_type "$1" \
    --cluster_inference_type "$2" --expression_inference_type "$3"
done
for stages in "model model lookup" "closest_slice majority_baseline lookup"; do
  set -- $stages
  run supp_cascade.csv "${CASCADE[@]}" --location_inference_type "$1" \
    --cluster_inference_type "$2" --expression_inference_type "$3" "${NO_GUIDANCE[@]}"
done
