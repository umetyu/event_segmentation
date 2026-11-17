#!/bin/bash
# Run from an Izar login shell; dataset files are read-only.
set -eo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: bash cluster/submit_dsec_preflight.sh DSEC_ROOT [SEQUENCE]' >&2
    exit 2
fi
if ! type module >/dev/null 2>&1; then
    echo 'Run this from an Izar login shell (bash -l).' >&2
    exit 2
fi
module purge
module load gcc/11.3.0 python/3.10.4
set -u
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="$(cd -- "$1" && pwd)"
export DSEC_ROOT="$data_root"
export DSEC_SEQUENCE="${2:-}"
venv_dir="${EVENT_FLOW_VENV:-$HOME/venvs/event-flow-torch251-cu121}"
export BENCHMARK_PYTHON="$venv_dir/bin/python"
if [[ ! -x "$BENCHMARK_PYTHON" ]]; then
    echo 'Run cluster/setup_izar.sh first to create the model environment.' >&2
    exit 2
fi
"$BENCHMARK_PYTHON" -m pip install --no-cache-dir --only-binary=:all: -r "$project_dir/requirements-data.txt"
"$BENCHMARK_PYTHON" -m pip check
"$BENCHMARK_PYTHON" "$project_dir/scripts/fetch_sources.py"
cd "$project_dir"
mkdir -p results
"$BENCHMARK_PYTHON" -m pip freeze > results/izar-dsec-environment.txt
submission="$(mktemp results/dsec-submission.XXXXXX)"
trap 'rm -f "$submission"' EXIT
status=0
# Slurm prints the job ID immediately; --wait returns the job's exit status.
echo 'Submitting DSEC preflight; waiting for the job to finish, then displaying its report.'
sbatch --wait --parsable --export=ALL cluster/dsec_preflight.sbatch | tee "$submission" || status=$?
job_id="$(head -n 1 "$submission")"
job_id="${job_id%%;*}"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
    if [[ -f "results/dsec-preflight-${job_id}.json" ]]; then
        cat "results/dsec-preflight-${job_id}.json"
    elif [[ -f "results/dsec-preflight-${job_id}.log" ]]; then
        cat "results/dsec-preflight-${job_id}.log"
    fi
fi
exit "$status"
