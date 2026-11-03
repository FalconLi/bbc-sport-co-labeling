"""Training, inference and evaluation commands over the same reusable model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .estimator import CoLabelingClassifier
from .io import load_dataset
from .metrics import evaluate_model


def _label(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train general multi-view co-labeling and save its final model.",
        epilog="NPZ features: view_0, view_1, ... in matching row order. SSL: y + known_mask. MIL: bag_ids + y (bag labels repeated per instance). ROD: known_normal_mask.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=["ssl", "mil", "rod"], required=True)
    parser.add_argument("--layers", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    parser.add_argument("--normalize", choices=["none", "l1", "l2"], default="none")
    parser.add_argument("--gamma", type=float, help="RBF gamma; default is fitted from training distances.")
    parser.add_argument("--c", type=float, default=10.)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--inner-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--outer-tolerance", type=float, default=1e-6)
    parser.add_argument("--outlier-fraction", type=float, default=.1, help="Prior ratio among unlabeled ROD samples.")
    parser.add_argument("--normal-label", type=_label, default="normal")
    parser.add_argument("--outlier-label", type=_label, default="outlier")
    parser.add_argument("--positive-label", type=_label, help="MIL positive bag label; default is the last sorted label.")
    parser.add_argument("--minimum-positive", type=int, default=1, help="Minimum positive instances per positive MIL bag.")
    parser.add_argument("--mil-positive-fraction", type=float, help="Require this fraction per positive MIL bag instead of a minimum count.")
    parser.add_argument("--proportion-offsets", nargs="+", type=float, help="SSL/ROD perturbations; default is -0.1 to +0.1 by 0.02.")
    parser.add_argument("--class-priors", nargs="+", type=float, help="Optional SSL proportions in sorted class order; default uses supplied labels.")
    parser.add_argument("--mil-biases", nargs="+", type=float, default=[-.5, -.4, -.3])
    parser.add_argument("--norms", nargs=3, type=float, metavar=("SOURCE", "BIAS", "ITERATION"), help="Optional nested norm exponents, e.g. inf 2 1.")
    args = parser.parse_args(argv)
    try:
        views, data = load_dataset(args.input)
        if args.task == "ssl" and ("y" not in data or "known_mask" not in data):
            raise ValueError("SSL training NPZ needs y and known_mask.")
        if args.task == "mil" and ("y" not in data or "bag_ids" not in data):
            raise ValueError("MIL training NPZ needs y and bag_ids.")
        if args.task == "rod" and "known_normal_mask" not in data:
            raise ValueError("ROD training NPZ needs known_normal_mask.")
        model = CoLabelingClassifier(
            task=args.task, layers=args.layers, c=args.c, max_rounds=args.max_rounds, max_inner=args.inner_iterations,
            tolerance=args.tolerance, outer_tolerance=args.outer_tolerance, kernel=args.kernel, gamma=args.gamma,
            normalization=None if args.normalize == "none" else args.normalize,
            outlier_fraction=args.outlier_fraction, normal_label=args.normal_label, outlier_label=args.outlier_label,
            positive_label=args.positive_label, minimum_positive=args.minimum_positive,
            mil_positive_fraction=args.mil_positive_fraction, proportion_offsets=args.proportion_offsets,
            mil_biases=args.mil_biases, norms=args.norms, class_priors=args.class_priors,
        )
        fit_data = ({"y": data["y"], "known_mask": data["known_mask"]} if args.task == "ssl" else
                    {"y": data["y"], "bag_ids": data["bag_ids"]} if args.task == "mil" else
                    {"known_normal_mask": data["known_normal_mask"]})
        with threadpool_limits(limits=1):
            model.fit(views, **fit_data, progress=lambda line: print(line, flush=True))
        model.save(args.model)
        summary = {"task": model.task, "views": model.n_views_, "classes": model.classes_.tolist(),
                   "rounds_per_binary_task": model.n_iter_, "outer_converged": model.outer_converged_,
                   "stop_reasons": model.stop_reasons_, "model": str(args.model)}
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.error(str(error))


def _inference_args(description, argv):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit", choices=["auto", "instance", "bag"], default="auto",
                        help="Default is bag for MIL files with bag_ids, otherwise instance.")
    return parser, parser.parse_args(argv)


def _use_bags(model, data, unit):
    bags = unit == "bag" or (unit == "auto" and model.task == "mil" and "bag_ids" in data)
    if bags and (model.task != "mil" or "bag_ids" not in data):
        raise ValueError("Bag predictions require a MIL model and input bag_ids.")
    return bags


def predict_main(argv=None):
    parser, args = _inference_args("Predict with a saved co-labeling model; no labels are needed.", argv)
    try:
        model = CoLabelingClassifier.load(args.model)
        views, data = load_dataset(args.input)
        if _use_bags(model, data, args.unit):
            bags, scores = model.bag_decision_function(views, data["bag_ids"])
            payload = {"bag_ids": bags, "scores": scores}
        else:
            scores_by_view = model.decision_function_views(views)
            scores = scores_by_view.mean(axis=0)
            payload = {"scores": scores, "view_scores": scores_by_view}
        payload.update(labels=model.classes_[scores.argmax(axis=1)], classes=model.classes_)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        print(f"Saved {len(scores)} predictions to {args.output}")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


def evaluate_main(argv=None):
    parser, args = _inference_args("Evaluate a saved model against test labels y in a separate NPZ.", argv)
    try:
        model = CoLabelingClassifier.load(args.model)
        views, data = load_dataset(args.input)
        if "y" not in data:
            raise ValueError("Evaluation needs true test labels y.")
        bags = data["bag_ids"] if _use_bags(model, data, args.unit) else None
        report = evaluate_model(model, views, data["y"], bag_ids=bags)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
