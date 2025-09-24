python main.py --data_mode "rq2" --location_inference_type "uniform_circle" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/rq2.csv" --metric_sampling 100
python main.py --data_mode "rq2" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/rq2.csv" --metric_sampling 100

python main.py --data_mode "rq1" --location_inference_type "closest_slice" --cluster_inference_type "majority_baseline" --expression_inference_type "lookup" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/debug.csv"
python main.py --data_mode "rq1" --location_inference_type "model" --cluster_inference_type "model" --expression_inference_type "model" --metrics "soft_correlation,soft_f1,soft_gene_distance" --out_csv "results/rq1_for_plot.csv"

