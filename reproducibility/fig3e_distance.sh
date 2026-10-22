#!/usr/bin/env bash
# Fig. 3E: soft Spearman at radius 0.15 as the reference sections move away from
# the held-out Zhuang-ABCA-2 sections. Distances 1-4 are the data modes
# rq3_v2_d1 ... rq3_v2_d4; distance 5 is rq3_v2 itself.
#
# As in the paper, the MIMYR line is the zero-shot model at distances 1-4 and the
# fine-tuned model at distance 5, and the lookup line is the baseline at all five.
#
# Output: $OUT/fig3e_distance.csv, 50 rows (5 distances x 2 methods x 5 sections).
# The distance-5 rows repeat the corresponding rows of fig3_rq3v2.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for mode in rq3_v2_d1 rq3_v2_d2 rq3_v2_d3 rq3_v2_d4; do
  run fig3e_distance.csv --pretrained zhuang1 --data_mode "$mode" \
    "${FULL_MODEL[@]}" --metrics "$EXPR_METRICS"
done
run fig3e_distance.csv --pretrained zhuang2-finetuned --data_mode rq3_v2 \
  "${FULL_MODEL[@]}" --metrics "$EXPR_METRICS"

for mode in rq3_v2_d1 rq3_v2_d2 rq3_v2_d3 rq3_v2_d4 rq3_v2; do
  run fig3e_distance.csv --pretrained zhuang1 --data_mode "$mode" --no-omit_x \
    "${BASELINE[@]}" --metrics "$EXPR_METRICS"
done
