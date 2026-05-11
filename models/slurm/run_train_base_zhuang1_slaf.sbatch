#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --gres gpu:3
#SBATCH --exclude oven-0-17
#SBATCH -c 64
#SBATCH --mem=450Gb
#SBATCH --job-name TG_base_slaf
#SBATCH --output /work/magroup/skrieger/tissue_generator/spencer_mimyr/models/slurm/logs/TG_base_slaf-%J.txt
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

# Step 1: Single-thread preprocessing — caches train.slaf to output_dir.
# No-op if the cache already exists, so it's safe to leave this in permanently.
python main.py --config configs/train_base_zhuang1_slaf.yaml --expression_preprocess_only

# Step 2: Multi-GPU training — loads from cache, skips slice loading entirely.
torchrun --nproc_per_node=3 --master_port $port main.py --config configs/train_base_zhuang1_slaf.yaml
