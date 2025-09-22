# python main.py --data_mode "rq1" --location_inference_type "closest_slice" --cluster_inference_type "end" --metrics "gridized_l1_distance,gridized_kl_divergence" --out_csv "results/location_ablation.csv"
# python main.py --data_mode "rq1" --location_inference_type "model" --cluster_inference_type "end" --metrics "gridized_l1_distance,gridized_kl_divergence" --out_csv "results/location_ablation.csv"


# python main.py --data_mode "rq1" --location_inference_type "skip" --cluster_inference_type "majority_baseline" --expression_inference_type "end" --metrics "soft_accuracy,delauney_colocalization" --out_csv "results/cluster_ablation.csv"
# python main.py --data_mode "rq1" --location_inference_type "skip" --cluster_inference_type "model" --expression_inference_type "end" --metrics "soft_accuracy,delauney_colocalization" --out_csv "results/cluster_ablation.csv"

python main.py --data_mode "rq1" --location_inference_type "skip" --cluster_inference_type "skip" --expression_inference_type "lookup" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/expression_ablation.csv"
python main.py --data_mode "rq1" --location_inference_type "skip" --cluster_inference_type "skip" --expression_inference_type "model" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/expression_ablation.csv"

# python main.py --data_mode "rq1_2" --location_inference_type "closest_slice" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/debug.csv"
# python main.py --data_mode "rq1_2" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/debug.csv"
