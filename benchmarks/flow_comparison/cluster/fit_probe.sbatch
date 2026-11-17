#!/bin/bash -l
#SBATCH --job-name=event-flow-fit
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=results/fit-probe-%j.log

set -eo pipefail
module purge
module load gcc/11.3.0 python/3.10.4
set -u
: "${BENCHMARK_PYTHON:?Launch through submit_fit_probe.sh}"
: "${SHORT_TRAIN_RUN:?Set SHORT_TRAIN_RUN}"
cd "${SLURM_SUBMIT_DIR:?}"
nvidia-smi
"$BENCHMARK_PYTHON" -m pip freeze > "results/fit-probe-environment-${SLURM_JOB_ID}.txt"
arguments=(--source-run "$SHORT_TRAIN_RUN" --device cuda)
if [[ -n "${FIT_PROBE_RESUME:-}" ]]; then
    arguments+=(--output "$FIT_PROBE_RESUME" --resume)
else
    arguments+=(--output "results/fit-probe-${SLURM_JOB_ID}")
fi
"$BENCHMARK_PYTHON" -u scripts/fit_probe.py "${arguments[@]}"
