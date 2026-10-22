#!/usr/bin/env bash
# Supplementary Fig. (parameters) E: expression decoding on Zhuang-ABCA-2
# (rq3_v2), with ground-truth locations and cell types. Two one-dimensional
# ladders around the default (top_k 5, margin 0.01):
#   top_k in {1, 5, 20, 50, off} at margin 0.01, and
#   margin in {0, 0.005, 0.05} at top_k 5 (margin 0 is multinomial sampling).
#
# Output: $OUT/supp_decoding.csv, 40 rows (8 settings x 5 test sections).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ORACLE=(--pretrained zhuang1 --data_mode rq3_v2
  --location_inference_type skip --cluster_inference_type skip
  --expression_inference_type model --metrics "$EXPR_METRICS")

for setting in "1 0.01" "5 0.01" "20 0.01" "50 0.01" "0 0.01" \
               "5 0" "5 0.005" "5 0.05"; do
  set -- $setting
  run supp_decoding.csv "${ORACLE[@]}" --top_k "$1" --expression_select_max_index_margin "$2"
done
