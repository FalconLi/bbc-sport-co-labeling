"""Train a model on any aligned multi-view feature dataset."""
import os

for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

from colabeling.cli import train_main

if __name__ == "__main__":
    train_main()
