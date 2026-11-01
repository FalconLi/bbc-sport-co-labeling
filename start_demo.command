#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "Please create .venv using the Setup section in README.md, then open this launcher again."
  exit 1
fi
source .venv/bin/activate
python run_demo.py
