#!/bin/bash
#SBATCH -p gpu-large
###SBATCH -w oven-0-13
#SBATCH --gres gpu:4
#SBATCH -c 8
#SBATCH --mem=300Gb
#SBATCH --job-name rq3_finetune
#SBATCH --output /work/magroup/skrieger/tissue_generator/spencer_mimyr/models/slurm/logs/rq3_finetune-%J.txt
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

torchrun --nproc_per_node=4 --master_port $port main.py --config configs/finetune_rq3and1.yaml
