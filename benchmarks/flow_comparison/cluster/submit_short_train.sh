#!/bin/bash
set -eo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: bash cluster/submit_short_train.sh DSEC_ROOT [CONFIG_JSON]' >&2
    exit 2
fi
if ! type module >/dev/null 2>&1; then
    echo 'Run from an Izar login shell (bash -l).' >&2
    exit 2
fi
module purge
module load gcc/11.3.0 python/3.10.4
set -u
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="$(cd -- "$1" && pwd)"
config_file="${2:-$project_dir/configs/short_train.json}"
config_file="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$config_file")"
export DSEC_ROOT="$data_root"
export SHORT_TRAIN_CONFIG="$config_file"
venv_dir="${EVENT_FLOW_VENV:-$HOME/venvs/event-flow-torch251-cu121}"
export BENCHMARK_PYTHON="$venv_dir/bin/python"
if [[ ! -x "$BENCHMARK_PYTHON" ]]; then
    echo 'Run cluster/setup_izar.sh first.' >&2
    exit 2
fi
"$BENCHMARK_PYTHON" -m pip install --no-cache-dir --only-binary=:all: -r "$project_dir/requirements-data.txt"
"$BENCHMARK_PYTHON" -m pip check
"$BENCHMARK_PYTHON" "$project_dir/scripts/fetch_sources.py"
cd "$project_dir"
mkdir -p results
submission="$(mktemp results/train-submission.XXXXXX)"
trap 'rm -f "$submission"' EXIT
status=0
echo 'Submitting both short training runs; waiting for completion and printing the summary.'
sbatch --wait --parsable --export=ALL cluster/short_train.sbatch | tee "$submission" || status=$?
job_id="$(head -n 1 "$submission")"
job_id="${job_id%%;*}"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
    if [[ -f "results/short-train-${job_id}/summary.json" ]]; then
        cat "results/short-train-${job_id}/summary.json"
    fi
    if [[ "$status" -ne 0 && -f "results/short-train-${job_id}.log" ]]; then
        tail -n 100 "results/short-train-${job_id}.log"
    fi
fi
exit "$status"
