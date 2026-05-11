#!/bin/bash
#SBATCH -p gpu-large
###SBATCH -w oven-0-13
#SBATCH --gres gpu:4
#SBATCH -c 16
#SBATCH --mem=450Gb
#SBATCH --job-name rq4_finetune
#SBATCH --output /work/magroup/skrieger/tissue_generator/spencer_mimyr/models/slurm/logs/rq4_finetune-%J.txt
#SBATCH -t 2-00:00:00

module load cuda-12.1
source /work/magroup/skrieger/newconda/etc/profile.d/conda.sh
conda activate scMulan2-slaf
cd /work/magroup/skrieger/tissue_generator/spencer_mimyr
mkdir -p /scratch/skrieger
export WANDB_DIR=/scratch/skrieger
export WANDB_API_KEY=$(cat ~/.wandb_api_key)

port=25902
while lsof -i TCP:$port &>/dev/null; do
  port=$((port + 1))
done

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_JOB_GPUS 
torchrun --nproc_per_node=4 --master_port $port main.py --config configs/finetune_rq4_rq3.yaml
