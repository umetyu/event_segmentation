#!/bin/bash -l
# Launch through cluster/setup_izar.sh, or export BENCHMARK_PYTHON before sbatch.
#SBATCH --job-name=event-flow-models
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/model-preflight-%j.log

set -eo pipefail
module purge
module load gcc/11.3.0 python/3.10.4 py-virtualenv
set -u
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${BENCHMARK_PYTHON:?Set BENCHMARK_PYTHON to the experiment virtualenv Python}"
cd "${SLURM_SUBMIT_DIR:?}"
nvidia-smi
"$BENCHMARK_PYTHON" scripts/model_preflight.py --device cuda --output "results/model-preflight-${SLURM_JOB_ID}.json"
