#!/bin/zsh
set -eu
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  task_python="$(command -v python || true)"
  if [[ -z "$task_python" ]]; then
    for candidate in /opt/homebrew/bin/python /usr/local/bin/python "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python"; do
      if [[ -x "$candidate" ]]; then task_python="$candidate"; break; fi
    done
  fi
  if [[ -z "$task_python" ]]; then
    print "Python is not available as 'python'. Install Python 3.12 or newer and make the python command available, then run this file again."
    exit 1
  fi
  "$task_python" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python download_data.py
python -m pytest -q
print "\nSetup is complete. Open start_demo.command to present."
