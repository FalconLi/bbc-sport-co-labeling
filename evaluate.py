"""Evaluate a saved model against separate test labels."""
import os

for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

from colabeling.cli import evaluate_main

if __name__ == "__main__":
    evaluate_main()
