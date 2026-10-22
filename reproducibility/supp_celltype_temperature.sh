#!/usr/bin/env bash
# Supplementary Fig. (parameters) F: cell-type sampling temperature T on the
# Zhuang-ABCA-2 distance-2 setting (rq3_v2_d2), with ground-truth locations.
#   T in {1, 0.001, 0.01}: sampling against two proxies for taking the mode
#                          -> $OUT/supp_celltype_argmax.csv (15 rows)
#   T in {1, 1.5, 2} x rare-type prior alpha in {0, 0.5, 1}
#                          -> $OUT/supp_celltype_sampling.csv (45 rows)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CELLTYPE=(--pretrained zhuang1 --data_mode rq3_v2_d2
  --location_inference_type skip --cluster_inference_type model
  --expression_inference_type model
  --metrics "$EXPR_METRICS,celltype_abundance,composition_entropy")

for T in 1 0.001 0.01; do
  run supp_celltype_argmax.csv "${CELLTYPE[@]}" --cluster_temperature "$T" --cluster_prior_alpha 0
done

for T in 1.0 1.5 2.0; do
  for A in 0.0 0.5 1.0; do
    run supp_celltype_sampling.csv "${CELLTYPE[@]}" --cluster_temperature "$T" --cluster_prior_alpha "$A"
  done
done
