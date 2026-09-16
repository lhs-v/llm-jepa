#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: bash setup_gemma4.sh [python-executable]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_exe="${1:-python3}"
venv_directory="$script_dir/.venv"
venv_python="$venv_directory/bin/python"

if [[ ! -x "$venv_python" ]]; then
    if [[ -e "$venv_directory" ]]; then
        echo "The existing .venv has no executable bin/python. Inspect it before recreating the environment." >&2
        exit 1
    fi
    "$python_exe" -m venv "$venv_directory"
fi

"$venv_python" -m pip install --no-cache-dir --disable-pip-version-check \
    'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
"$venv_python" -m pip install --no-cache-dir --disable-pip-version-check \
    -r "$script_dir/requirements-gemma4.txt"
"$venv_python" -m pip check
"$venv_python" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"

echo "Environment ready: $venv_python"
echo "See GEMMA4.md for data preparation, a short training check, and experiment commands."
