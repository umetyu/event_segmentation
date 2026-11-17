#!/bin/bash
# Run from an Izar login shell: bash cluster/setup_izar.sh
# Only binary packages are installed here; GPU execution is submitted via Slurm.
set -eo pipefail

if ! type module >/dev/null 2>&1; then
    echo 'Run this from an Izar login shell (bash -l) so the module command is available.' >&2
    exit 2
fi
module purge
module load gcc/11.3.0 python/3.10.4
set -u

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${EVENT_FLOW_VENV:-/scratch/$(id -un)/event-flow-benchmark/venv-torch251-cu121}"
python -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12), sys.version; print(sys.version)'

if [[ ! -e "$venv_dir" ]]; then
    mkdir -p "$(dirname -- "$venv_dir")"
    python -m venv "$venv_dir"
elif [[ ! -f "$venv_dir/pyvenv.cfg" || ! -x "$venv_dir/bin/python" ]]; then
    echo "Existing path is not a valid virtualenv: $venv_dir" >&2
    exit 2
fi

# Older venv-seeded pip versions reject normalized names in the PyTorch index.
"$venv_dir/bin/python" -m pip install --no-cache-dir --only-binary=:all: --upgrade 'pip==25.0.1'
"$venv_dir/bin/python" -m pip install --no-cache-dir --only-binary=:all: 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
"$venv_dir/bin/python" -m pip install --no-cache-dir --only-binary=:all: 'numpy==1.26.4'
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" "$project_dir/scripts/fetch_sources.py"

mkdir -p "$project_dir/results"
"$venv_dir/bin/python" -m pip freeze > "$project_dir/results/izar-environment.txt"
module -t list > "$project_dir/results/izar-modules.txt" 2>&1
export BENCHMARK_PYTHON="$venv_dir/bin/python"
cd "$project_dir"
sbatch --export=ALL cluster/model_preflight.sbatch
