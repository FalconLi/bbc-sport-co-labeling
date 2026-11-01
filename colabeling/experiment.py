"""Predeclared splits and metrics; evaluation is separate from all learners."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from threadpoolctl import threadpool_limits

from .data import ARCHIVES, CATEGORIES, BBCSport, make_split
from .models import fit_colabeling, fit_cotraining, fit_supervised

METHODS = ["supervised", "cotraining", "colabel_flat", "colabel_grouped"]
METHOD_NAMES = {
    "supervised": "Known answers only", "cotraining": "Co-training",
    "colabel_flat": "Co-labeling: one layer", "colabel_grouped": "Co-labeling: two layers",
}
PAPER = {
    "source": "Xu et al., TPAMI 2016, Table 3, BBCSport, View1+View2",
    "doi": "10.1109/TPAMI.2015.2476813",
    "map": {"supervised": 80.65, "cotraining": 90.31, "colabel_flat": 94.81, "colabel_grouped": 95.44},
    "std": {"supervised": 3.38, "cotraining": 4.89, "colabel_flat": 1.41, "colabel_grouped": 1.63},
    "runs": 10,
}


@dataclass
class ExperimentConfig:
    runs: int = 10
    seed: int = 0
    labels_per_class: int = 2
    c: float = 10.0
    rounds: int = 10
    max_inner: int = 100
    tolerance: float = 1e-5


def evaluate(answers, view_scores):
    scores = np.mean(view_scores, axis=0)
    per_class = [average_precision_score(answers == category, scores[:, category]) * 100 for category in range(5)]
    return {
        "map": float(np.mean(per_class)), "ap_by_class": [float(x) for x in per_class],
        "accuracy": float(accuracy_score(answers, scores.argmax(axis=1)) * 100),
        "map_by_view": [float(np.mean([
            average_precision_score(answers == category, view[:, category]) * 100 for category in range(5)
        ])) for view in view_scores],
    }


def run_experiment(dataset: BBCSport, config: ExperimentConfig, output_dir: Path):
    if config.runs < 1 or config.rounds < 1:
        raise ValueError("At least one run and one learning round are required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    results, first_recording = [], None
    with threadpool_limits(limits=1):
        for run_index in range(config.runs):
            seed = config.seed + run_index
            started = time.perf_counter()
            split = make_split(dataset, seed, config.labels_per_class)
            run_metrics, recordings = {}, {}
            for method in METHODS:
                print(f"Run {run_index + 1}/{config.runs}, seed {seed}: {METHOD_NAMES[method]}", flush=True)
                if method == "supervised":
                    learned = fit_supervised(split.train_views, split.test_views, split.known_classes, c=config.c)
                elif method == "cotraining":
                    learned = fit_cotraining(split.train_views, split.test_views, split.known_classes, c=config.c, rounds=config.rounds)
                else:
                    learned = fit_colabeling(
                        split.train_views, split.test_views, split.known_classes, c=config.c, rounds=config.rounds,
                        grouped=method == "colabel_grouped", max_inner=config.max_inner,
                        tolerance=config.tolerance,
                        progress=lambda text: print(f"  {text}", flush=True),
                    )
                # Hidden answers are read only after fitting; no tuning uses them.
                metrics = evaluate(dataset.labels[split.test], learned.test_scores[-1])
                diagnostics = [view for detail in learned.details for cat in detail.get("categories", [])
                               if detail["round"] <= cat["completed_rounds"] for view in cat["views"]]
                metrics["solver"] = {
                    "fits": len(diagnostics), "converged": sum(d["converged"] for d in diagnostics),
                    "max_relative_gap": max((d["relative_gap"] for d in diagnostics), default=0.0),
                }
                run_metrics[method] = metrics
                print(f"  test ranking score {metrics['map']:.2f}; accuracy {metrics['accuracy']:.2f}%", flush=True)
                if run_index == 0:
                    recordings[method] = {
                        "train_scores": [np.round(x, 8).tolist() for x in learned.train_scores],
                        "test_scores": [np.round(x, 8).tolist() for x in learned.test_scores],
                        "details": learned.details,
                        "round_metrics": [evaluate(dataset.labels[split.test], x) for x in learned.test_scores],
                    }
            record = {"seed": seed, "seconds": time.perf_counter() - started, "methods": run_metrics}
            results.append(record)
            (output_dir / f"seed-{seed}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            if run_index == 0:
                first_recording = {
                    "seed": seed, "train_indices": split.train.tolist(), "test_indices": split.test.tolist(),
                    "known_mask": split.known_mask.tolist(), "known_classes": split.known_classes.tolist(),
                    "word_indices": [x.tolist() for x in split.word_indices], "methods": recordings,
                }
                (output_dir / "recording.json").write_text(json.dumps(first_recording, separators=(",", ":"), allow_nan=False))
    summary = {
        method: {
            "map_mean": float(np.mean([r["methods"][method]["map"] for r in results])),
            "map_std": float(np.std([r["methods"][method]["map"] for r in results], ddof=1)) if len(results) > 1 else None,
            "accuracy_mean": float(np.mean([r["methods"][method]["accuracy"] for r in results])),
        } for method in METHODS
    }
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "seeds": [r["seed"] for r in results],
        "dataset": {"name": "BBCSport", "articles": 737, "words": 4613, "train": 370, "test": 367, "archive_sha256": ARCHIVES},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in ["numpy", "scipy", "scikit-learn", "osqp"]}},
        "methods": summary, "runs": results, "paper": PAPER,
        "notes": [
            "Independent reimplementation; original author code and seeds unavailable.",
            "C, stopping tolerances, and co-training selection policy are explicit implementation choices.",
            "All C values are fixed before evaluation; held-out answers never enter training or parameter selection.",
            "The first predeclared seed supplies the demo recording; it is not chosen for its result.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report, first_recording
