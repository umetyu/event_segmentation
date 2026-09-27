#!/bin/bash
#SBATCH --job-name=ev-layerseg
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail

module purge
module load gcc/11.3.0
module load python/3.10.4
module load cuda/11.8.0

source /home/umetsu/venvs/event-segmentation/bin/activate
cd /home/umetsu/projects/event_segmentation

mkdir -p logs mlruns

python -u train_flow.py \
    --config configs/train_izar.yml \
    --path_mlflow file:///home/umetsu/projects/event_segmentation/mlruns
