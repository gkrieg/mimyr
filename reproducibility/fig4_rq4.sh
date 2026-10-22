#!/usr/bin/env bash
# Fig. 4C: soft Spearman on five held-out sagittal Zhuang-ABCA-3 sections (data
# mode rq4), MIMYR fine-tuned on the remaining Zhuang-ABCA-3 sections against the
# lookup baseline.
#
# Output: $OUT/fig4_rq4.csv, 10 rows (2 methods x 5 test sections).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# zhuang3-finetuned applies reference guidance 50 (not 75) and --metric_filter_by_gt.
# Its location and cell-type models are the MERSCOPE ones: the Zhuang-ABCA-1
# location model is coronal and does not apply to sagittal sections.
run fig4_rq4.csv --pretrained zhuang3-finetuned --data_mode rq4 \
  "${FULL_MODEL[@]}" --metrics "$EXPR_METRICS"

run fig4_rq4.csv --pretrained zhuang3-finetuned --data_mode rq4 \
  "${BASELINE[@]}" --metrics "$EXPR_METRICS"
