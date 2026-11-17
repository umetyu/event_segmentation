#!/bin/bash -l
#SBATCH --job-name=event-flow-diagnose
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/diagnose-%j.log

set -eo pipefail
module purge
module load gcc/11.3.0 python/3.10.4
set -u
: "${BENCHMARK_PYTHON:?Launch through submit_diagnostics.sh}"
: "${SHORT_TRAIN_RUN:?Set SHORT_TRAIN_RUN}"
cd "${SLURM_SUBMIT_DIR:?}"
nvidia-smi
"$BENCHMARK_PYTHON" -m pip freeze > "results/diagnose-environment-${SLURM_JOB_ID}.txt"
"$BENCHMARK_PYTHON" -u scripts/diagnose_short_run.py \
    --run "$SHORT_TRAIN_RUN" --device cuda \
    --output "results/diagnose-${SLURM_JOB_ID}"
