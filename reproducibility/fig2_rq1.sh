#!/usr/bin/env bash
# Fig. 2C: soft Spearman on the five held-out MERSCOPE sections (data mode rq1),
# MIMYR against the rule-based lookup baseline.
#
# Output: $OUT/fig2_rq1.csv, 10 rows (2 methods x 5 test sections).
# SpatialZ, the third bar, is a separate method and is not run here.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# MIMYR. The merscope model applies reference guidance 75 and --metric_filter_by_gt.
run fig2_rq1.csv --pretrained merscope --data_mode rq1 \
  "${FULL_MODEL[@]}" --metrics "$EXPR_METRICS"

run fig2_rq1.csv --pretrained merscope --data_mode rq1 \
  "${BASELINE[@]}" --metrics "$EXPR_METRICS"
