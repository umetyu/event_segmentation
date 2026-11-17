#!/bin/bash -l
#SBATCH --job-name=event-flow-dsec
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/dsec-preflight-%j.log

set -eo pipefail
module purge
module load gcc/11.3.0 python/3.10.4
set -u
: "${BENCHMARK_PYTHON:?Launch through cluster/submit_dsec_preflight.sh}"
: "${DSEC_ROOT:?Set DSEC_ROOT}"
cd "${SLURM_SUBMIT_DIR:?}"
nvidia-smi
args=(--root "$DSEC_ROOT" --samples 3 --snn-steps 5 --device cuda)
if [[ -n "${DSEC_SEQUENCE:-}" ]]; then
    args+=(--sequence "$DSEC_SEQUENCE")
fi
"$BENCHMARK_PYTHON" scripts/dsec_preflight.py "${args[@]}" --output "results/dsec-preflight-${SLURM_JOB_ID}.json"
