python main.py --data_mode "rq1" --location_model_checkpoint "model_checkpoints/best_model_intra2hole.pt" --location_inference_type "model" --cluster_inference_type "end" 
python main.py --data_mode "rq1" --location_model_checkpoint "model_checkpoints/best_model_intra2hole.pt" --location_inference_type "neighborhood_sampling" --cluster_inference_type "end" 


python main.py --data_mode "rq1" --location_model_checkpoint "model_checkpoints/best_model_intra2hole.pt" --location_inference_type "skip" --cluster_inference_type "neighborhood_majority" --expression_inference_type "end"
python main.py --data_mode "rq1" --location_model_checkpoint "model_checkpoints/best_model_intra2hole.pt" --location_inference_type "skip" --cluster_inference_type "model_homogenize_subclass" --expression_inference_type "end"

python main.py --data_mode "rq1" --location_model_checkpoint "model_checkpoints/best_model_intra2hole.pt" --location_inference_type "skip" --cluster_inference_type "skip" --expression_inference_type "lookup"
python main.py --data_mode "rq1" --location_model_checkpoint "model_checkpoints/best_model_intra2hole.pt" --location_inference_type "skip" --cluster_inference_type "skip" --expression_inference_type "model"
