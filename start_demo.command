#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "Please run setup.command once before opening the demo."
  exit 1
fi
source .venv/bin/activate
python run_demo.py
