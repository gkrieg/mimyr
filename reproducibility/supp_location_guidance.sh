#!/usr/bin/env bash
# Supplementary Fig. (parameters) C, D: KDE backward guidance for the location
# model on Zhuang-ABCA-2 (rq3_v2). Strength eta in {0, 0.005, 0.01, 0.05, 0.1} at
# bandwidth h = 0.01, and h in {0.005, 0.02, 0.05} at eta = 0.01, plus the
# closest-slice baseline. Only the location stage runs (`--cluster_inference_type end`).
#
# Output: $OUT/supp_location_guidance.csv, 45 rows (9 settings x 5 test sections).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOCATION_ONLY=(--pretrained zhuang1 --data_mode rq3_v2 --cluster_inference_type end --expression_inference_type end
  --metrics gridized_l1_distance,gridized_kl_divergence)

for setting in "0 0.01" "0.005 0.01" "0.01 0.01" "0.05 0.01" "0.1 0.01" \
               "0.01 0.005" "0.01 0.02" "0.01 0.05"; do
  set -- $setting
  run supp_location_guidance.csv "${LOCATION_ONLY[@]}" --location_inference_type model \
    --guidance_signal "$1" --kde_bandwidth "$2"
done
run supp_location_guidance.csv "${LOCATION_ONLY[@]}" --location_inference_type closest_slice
