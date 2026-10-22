#!/usr/bin/env bash
# Supplementary Fig. (ablations) B-D: each stage on its own, with the ground truth
# for every earlier stage, at reference distances 1-4 on Zhuang-ABCA-2
# (rq3_v2_d1 ... rq3_v2_d4), MIMYR (zero-shot) against the rule-based alternative.
#   stage 1  location               -> $OUT/supp_stage_location.csv   (40 rows)
#   stage 2  cell type, true locations
#                                   -> $OUT/supp_stage_celltype.csv   (40 rows)
#   stage 3  expression, true locations and cell types
#                                   -> $OUT/supp_stage_expression.csv (40 rows)
# Stage 2 is reported at the class level. main.py scores cell-type tokens, so
# analysis/celltype_class_accuracy.py rescores the stage-2 sections afterwards.
# SpatialZ, the third method in these panels, is not run here.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NO_GUIDANCE=(--no-expression_weak_teacher_forcing
  --wtf_strength_gene 0 --wtf_strength_bin 0 --wtf_strength_value 0)

for d in 1 2 3 4; do
  mode=(--pretrained zhuang1 --data_mode "rq3_v2_d$d")

  for loc in model closest_slice; do
    run supp_stage_location.csv "${mode[@]}" --location_inference_type "$loc" \
      --cluster_inference_type end --expression_inference_type end \
      --metrics gridized_l1_distance,gridized_kl_divergence
  done

  for ct in model majority_baseline; do
    run supp_stage_celltype.csv "${mode[@]}" --location_inference_type skip \
      --cluster_inference_type "$ct" --expression_inference_type end \
      --metrics soft_accuracy,celltype_abundance
  done

  run supp_stage_expression.csv "${mode[@]}" --location_inference_type skip \
    --cluster_inference_type skip --expression_inference_type model \
    --metrics "$EXPR_METRICS"
  run supp_stage_expression.csv "${mode[@]}" --location_inference_type skip \
    --cluster_inference_type skip --expression_inference_type lookup \
    --metrics "$EXPR_METRICS" "${NO_GUIDANCE[@]}"
done

[ -n "${DRY_RUN:-}" ] || "$PYTHON" reproducibility/analysis/celltype_class_accuracy.py \
  --csv "$OUT/supp_stage_celltype.csv" --data_dir "$DATA_DIR" \
  --out_csv "$OUT/supp_stage_celltype_class.csv"
