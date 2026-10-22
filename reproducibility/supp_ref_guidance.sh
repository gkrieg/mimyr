#!/usr/bin/env bash
# Supplementary Fig. (parameters) A, B: reference-guidance strength s in
# {0, 25, 50, 75, 100} on both atlases, plus the lookup baseline as the s -> 100
# limit. Locations and cell types are the ground truth (`skip`), so only the
# expression model is tested.
#
# Output: $OUT/supp_ref_guidance_rq1.csv and $OUT/supp_ref_guidance_rq3.csv,
# 30 rows each (6 settings x 5 test sections).
# The per-cell KS values in the panels come from analysis/per_cell_ks.py.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ORACLE=(--location_inference_type skip --cluster_inference_type skip)
NO_GUIDANCE=(--no-expression_weak_teacher_forcing
  --wtf_strength_gene 0 --wtf_strength_bin 0 --wtf_strength_value 0)

for setting in "rq1 merscope" "rq3 zhuang1"; do
  set -- $setting
  csv="supp_ref_guidance_$1.csv"
  mode=$([ "$1" = rq1 ] && echo rq1 || echo rq3_v2)

  run "$csv" --pretrained "$2" --data_mode "$mode" "${ORACLE[@]}" \
    --expression_inference_type model --metrics "$EXPR_METRICS" "${NO_GUIDANCE[@]}"
  for s in 25 50 75 100; do
    run "$csv" --pretrained "$2" --data_mode "$mode" "${ORACLE[@]}" \
      --expression_inference_type model --metrics "$EXPR_METRICS" $(guidance "$s")
  done
  run "$csv" --pretrained "$2" --data_mode "$mode" "${ORACLE[@]}" \
    --expression_inference_type lookup --metrics "$EXPR_METRICS" "${NO_GUIDANCE[@]}"
done
