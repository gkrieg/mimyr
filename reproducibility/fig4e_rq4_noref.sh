#!/usr/bin/env bash
# No-reference reconstruction of the held-out sagittal Zhuang-ABCA-3 sections (data
# mode rq4_noref): no Zhuang-ABCA-3 section is used, and the reference pool is all
# of Zhuang-ABCA-1, a different brain. The expression model is the Zhuang-ABCA-1
# base model with no fine-tuning, at reference guidance 50.
#
# Three runs, each writing its own CSV:
#   arm A   MERSCOPE location and cell-type models    -> $OUT/fig4e_noref_armA.csv
#   arm B   MERSCOPE location, Zhuang-1 cell types    -> $OUT/fig4e_noref_armB.csv
#   lookup  arm A's locations and cell types, lookup expression
#                                                     -> $OUT/fig4e_noref_lookup.csv
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# The zhuang1 model's own location model is coronal, so the sagittal sections use
# the MERSCOPE one.
MERSCOPE_LOCATION="$(hub_ckpt merscope location)"
MERSCOPE_CELLTYPE="$(hub_ckpt merscope celltype)"
NOREF=(--pretrained zhuang1 --data_mode rq4_noref --no-omit_x
  --location_model_checkpoint "$MERSCOPE_LOCATION"
  --metrics "$EXPR_METRICS")

run fig4e_noref_armA.csv "${NOREF[@]}" "${FULL_MODEL[@]}" $(guidance 50) \
  --cluster_model_checkpoint "$MERSCOPE_CELLTYPE"

run fig4e_noref_armB.csv "${NOREF[@]}" "${FULL_MODEL[@]}" $(guidance 50)

run fig4e_noref_lookup.csv "${NOREF[@]}" \
  --location_inference_type model --cluster_inference_type model \
  --expression_inference_type lookup \
  --cluster_model_checkpoint "$MERSCOPE_CELLTYPE" \
  --no-expression_weak_teacher_forcing \
  --wtf_strength_gene 0 --wtf_strength_bin 0 --wtf_strength_value 0
