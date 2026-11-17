#!/bin/bash -l
#SBATCH --job-name=event-flow-train
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=results/short-train-%j.log

set -eo pipefail
module purge
module load gcc/11.3.0 python/3.10.4
set -u
: "${BENCHMARK_PYTHON:?Launch through submit_short_train.sh}"
: "${DSEC_ROOT:?Set DSEC_ROOT}"
cd "${SLURM_SUBMIT_DIR:?}"
nvidia-smi
"$BENCHMARK_PYTHON" -m pip freeze > "results/short-train-environment-${SLURM_JOB_ID}.txt"
"$BENCHMARK_PYTHON" -u scripts/run_short_comparison.py \
    --root "$DSEC_ROOT" --device cuda --config "$SHORT_TRAIN_CONFIG" \
    --output "results/short-train-${SLURM_JOB_ID}"
