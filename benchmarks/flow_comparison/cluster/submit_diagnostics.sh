#!/bin/bash
set -eo pipefail
if [[ $# -ne 1 ]]; then
    echo 'Usage: bash cluster/submit_diagnostics.sh SHORT_TRAIN_RUN_DIRECTORY' >&2
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
export SHORT_TRAIN_RUN="$(cd -- "$1" && pwd)"
for file in data/manifest.json segmentation/last.pt snn/last.pt; do
    if [[ ! -f "$SHORT_TRAIN_RUN/$file" ]]; then
        echo "Missing existing run file: $SHORT_TRAIN_RUN/$file" >&2
        exit 2
    fi
done
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
submission="$(mktemp results/diagnose-submission.XXXXXX)"
trap 'rm -f "$submission"' EXIT
status=0
echo 'Submitting checkpoint diagnostics; waiting for completion and printing the summary.'
sbatch --wait --parsable --export=ALL cluster/diagnose_short_run.sbatch | tee "$submission" || status=$?
job_id="$(head -n 1 "$submission")"
job_id="${job_id%%;*}"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
    if [[ -f "results/diagnose-${job_id}/summary.json" ]]; then
        cat "results/diagnose-${job_id}/summary.json"
    fi
    if [[ "$status" -ne 0 && -f "results/diagnose-${job_id}.log" ]]; then
        tail -n 100 "results/diagnose-${job_id}.log"
    fi
fi
exit "$status"
