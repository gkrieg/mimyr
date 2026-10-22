#!/usr/bin/env bash
# Fig. 3C: soft Spearman on five held-out coronal Zhuang-ABCA-2 sections (data mode
# rq3_v2): the lookup baseline, MIMYR zero-shot (the Zhuang-ABCA-1 base model) and
# MIMYR fine-tuned on the remaining Zhuang-ABCA-2 sections.
#
# Output: $OUT/fig3_rq3v2.csv, 15 rows (3 methods x 5 test sections).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Both Zhuang models apply reference guidance 75, --metric_filter_by_gt and --omit_x.
run fig3_rq3v2.csv --pretrained zhuang1 --data_mode rq3_v2 \
  "${FULL_MODEL[@]}" --metrics "$EXPR_METRICS"

run fig3_rq3v2.csv --pretrained zhuang2-finetuned --data_mode rq3_v2 \
  "${FULL_MODEL[@]}" --metrics "$EXPR_METRICS"

run fig3_rq3v2.csv --pretrained zhuang1 --data_mode rq3_v2 --no-omit_x \
  "${BASELINE[@]}" --metrics "$EXPR_METRICS"
