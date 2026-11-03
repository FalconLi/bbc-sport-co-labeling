"""Load a saved model and predict new samples without retraining."""
import os

for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

from colabeling.cli import predict_main

if __name__ == "__main__":
    predict_main()
