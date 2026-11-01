"""Train/reproduce BBCSport; presentation itself uses the saved recording."""

import os

# Apple Accelerate is not controlled by threadpoolctl. Tiny repeated QPs are
# faster and more predictable with one math-library thread on presentation Macs.
for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import argparse
import json
from pathlib import Path

from colabeling.data import ROOT, load_data
from colabeling.experiment import ExperimentConfig, run_experiment
from colabeling.export import export_demo


def main():
    parser = argparse.ArgumentParser(description="Run the independent BBCSport experiment.")
    parser.add_argument("--runs", type=int, default=10, help="Number of successive predeclared seeds (default: 10).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels-per-class", type=int, default=2)
    parser.add_argument("--c", type=float, default=10.0, help="Fixed before evaluation; the paper does not supply this value.")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--inner-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "experiment")
    parser.add_argument("--export-demo", action="store_true", help="Replace the demo recording with this experiment's actual results.")
    args = parser.parse_args()
    if args.export_demo and args.labels_per_class != 2:
        parser.error('The ten-answer presentation demo requires --labels-per-class 2. Other budgets can be evaluated without --export-demo.')
    dataset = load_data()
    config = ExperimentConfig(args.runs, args.seed, args.labels_per_class, args.c, args.max_rounds, args.inner_iterations, args.tolerance)
    report, recording = run_experiment(dataset, config, args.output)
    print("\nMeasured results (ranking score out of 100):")
    for method, result in report["methods"].items():
        variability = f" +/- {result['map_std']:.2f}" if result["map_std"] is not None else " (one run)"
        print(f"  {method:20s} {result['map_mean']:.2f}{variability}")
    if args.export_demo:
        export_demo(dataset, report, recording)
    print(f"\nDetailed report: {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
