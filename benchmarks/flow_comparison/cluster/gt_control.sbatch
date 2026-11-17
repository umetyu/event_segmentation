#!/bin/bash -l
#SBATCH --job-name=event-flow-gt
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=results/gt-control-%j.log

set -eo pipefail
module purge
module load gcc/11.3.0 python/3.10.4
set -u
: "${BENCHMARK_PYTHON:?Launch through submit_gt_control.sh}"
: "${FIT_PROBE_RUN:?Set FIT_PROBE_RUN}"
cd "${SLURM_SUBMIT_DIR:?}"
nvidia-smi
"$BENCHMARK_PYTHON" -m pip freeze > "results/gt-control-environment-${SLURM_JOB_ID}.txt"
arguments=(--source-run "$FIT_PROBE_RUN" --device cuda)
if [[ -n "${GT_CONTROL_RESUME:-}" ]]; then
    arguments+=(--output "$GT_CONTROL_RESUME" --resume)
else
    arguments+=(--output "results/gt-control-${SLURM_JOB_ID}")
fi
"$BENCHMARK_PYTHON" -u scripts/gt_control.py "${arguments[@]}"
