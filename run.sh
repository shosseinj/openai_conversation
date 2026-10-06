#!/usr/bin/env bash
set -e
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$project_dir/.venv" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo 'Install Python 3.12 first, then run this launcher again.' >&2
        exit 1
    fi
    echo 'Creating .venv for this fresh checkout...'
    if ! python3 -m venv "$project_dir/.venv"; then
        echo 'Cannot create .venv. On Ubuntu, install python3-venv and try again.' >&2
        exit 1
    fi
fi
if [[ ! -x "$project_dir/.venv/bin/python" || ! -f "$project_dir/.venv/bin/activate" ]]; then
    echo 'Existing .venv is incomplete or belongs to another OS. Restore it or use a separate checkout; it will not be replaced.' >&2
    exit 1
fi
source "$project_dir/.venv/bin/activate"
exec "$project_dir/.venv/bin/python" "$project_dir/run.py" "$@"
