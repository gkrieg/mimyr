#!/bin/bash
#SBATCH -p gpu-large
###SBATCH -w oven-0-27
#SBATCH --gres gpu:3
#SBATCH -c 12
#SBATCH --mem=450Gb
#SBATCH --job-name TG_base
#SBATCH --output TG_base-%J.txt
#SBATCH -t 2-00:00:00

module load cuda-12.1
source /work/magroup/skrieger/newconda/etc/profile.d/conda.sh
conda activate scMulan2
cd /work/magroup/skrieger/tissue_generator/spencer_mimyr
mkdir -p /scratch/skrieger
export WANDB_DIR=/scratch/skrieger
export WANDB_API_KEY=$(cat ~/.wandb_api_key)

port=25902
while lsof -i TCP:$port &>/dev/null; do
  port=$((port + 1))
done

torchrun --nproc_per_node=3 --master_port $port main.py \
  --run_mode train \
  --data_mode rq1 \
  --data_dir /work/magroup/skrieger/tissue_generator/quantized_slices \
  --meta_info /work/magroup/skrieger/tissue_generator/MERFISH_aging/4hierarchy_metainfo_mouse_geneunion2_DAG.pt \
  --device cuda \
  --expression_output_dir /compute/oven-0-13/skrieger/TG-base-noreweight-test \
  --expression_adata2 /work/magroup/skrieger/tissue_generator/MERFISH_aging/sc_ref_xyz_mousegenes_raw.h5ad \
  --expression_batch_size 128 \
  --expression_max_len 500 \
  --expression_no_shuffle \
  --expression_epochs 1000 \
  --expression_lr 0.001 \
  --expression_lambda_val 0.1 \
  --expression_new_expression_size 100 \
  --expression_save_frequency 1 \
  --expression_xyz_noise \
  --expression_epoch_samples 1000000 \
  --expression_log_per_steps 10 \
  --skip_combined_fit \
  --expression_from_finetuned \
  --expression_rebalance_only \
  --expression_model_checkpoint /compute/oven-0-13/skrieger/TG-base-noreweight2/epoch36_model.pt
  #--expression_model_size medium
