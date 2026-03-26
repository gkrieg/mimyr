#!/bin/bash
#SBATCH -p gpu-large
###SBATCH -w oven-0-13
#SBATCH --gres gpu:2
#SBATCH -c 8
#SBATCH --mem=150Gb
#SBATCH --job-name rq3_finetune
#SBATCH --output rq3_finetune-%J.txt
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

torchrun --nproc_per_node=2 --master_port $port main.py \
  --run_mode train \
  --skip_combined_fit \
  --data_mode rq3_v2 \
  --data_dir /work/magroup/skrieger/tissue_generator/spencer_mimyr/data \
  --meta_info /work/magroup/skrieger/tissue_generator/MERFISH_aging/4hierarchy_metainfo_mouse_geneunion2_DAG.pt \
  --device cuda \
  --expression_output_dir /compute/oven-0-13/skrieger/Zhuang-2-v2 \
  --expression_batch_size 128 \
  --expression_max_len 500 \
  --expression_no_shuffle \
  --expression_epochs 1000 \
  --expression_lr 0.0005 \
  --expression_lambda_val 0.1 \
  --expression_new_expression_size 100 \
  --expression_save_frequency 2 \
  --expression_xyz_noise \
  --expression_from_finetuned \
  --expression_model_checkpoint /compute/oven-0-13/skrieger/TG-base-noreweight2/epoch100_model.pt
  #--expression_model_size medium
