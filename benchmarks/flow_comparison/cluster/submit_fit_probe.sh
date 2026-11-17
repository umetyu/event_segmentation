#!/bin/bash
set -eo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: bash cluster/submit_fit_probe.sh SHORT_TRAIN_RUN [FIT_PROBE_RUN_TO_RESUME]' >&2
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
source_run="$(cd -- "$1" && pwd)"
export SHORT_TRAIN_RUN="$source_run"
export FIT_PROBE_RESUME=''
if [[ $# -eq 2 ]]; then
    resume_run="$(cd -- "$2" && pwd)"
    export FIT_PROBE_RESUME="$resume_run"
fi
if [[ ! -f "$SHORT_TRAIN_RUN/data/manifest.json" ]]; then
    echo "Missing original data manifest: $SHORT_TRAIN_RUN/data/manifest.json" >&2
    exit 2
fi
venv_dir="${EVENT_FLOW_VENV:-$HOME/venvs/event-flow-torch251-cu121}"
export BENCHMARK_PYTHON="$venv_dir/bin/python"
if [[ ! -x "$BENCHMARK_PYTHON" ]]; then
    echo 'Run cluster/setup_izar.sh first.' >&2
    exit 2
fi
"$BENCHMARK_PYTHON" -m pip check
"$BENCHMARK_PYTHON" "$project_dir/scripts/fetch_sources.py"
cd "$project_dir"
mkdir -p results
submission="$(mktemp results/fit-submission.XXXXXX)"
trap 'rm -f "$submission"' EXIT
status=0
echo 'Submitting two-sample learning probe; waiting for completion and printing the learning curves.'
sbatch --wait --parsable --export=ALL cluster/fit_probe.sbatch | tee "$submission" || status=$?
job_id="$(head -n 1 "$submission")"
job_id="${job_id%%;*}"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
    output_dir="${FIT_PROBE_RESUME:-$project_dir/results/fit-probe-${job_id}}"
    echo "Results: $output_dir"
    if [[ -f "$output_dir/summary.json" ]]; then
        "$BENCHMARK_PYTHON" scripts/fit_probe_report.py "$output_dir"
    fi
    if [[ "$status" -ne 0 && -f "results/fit-probe-${job_id}.log" ]]; then
        tail -n 100 "results/fit-probe-${job_id}.log"
        echo "To resume, pass '$output_dir' as the second argument to submit_fit_probe.sh."
    fi
fi
exit "$status"
