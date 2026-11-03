"""Prepare an explanatory replay by observing, never modifying, the learner."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from colabeling.data import ROOT, load_data, make_split
from colabeling.solver import fit_mkl, project_labels

PREFIX = "window.BBCSPORT_STORY = "
COLORS = ["#289F8A", "#5083BE", "#D4AA43", "#C77864", "#9978B4"]
CHAPTERS = [
    ("initial", 0, 12, "Start with two trained readers"),
    ("first", 12, 18, "First exchange: predictions change"),
    ("sheets", 18, 35, "Round 2: create, exchange and keep sheets"),
    ("weights", 35, 60, "Learn how much influence each sheet gets"),
    ("updated", 60, 70, "Round 2: update the predictions"),
    ("repeat", 70, 85, "Repeat the process through round 8"),
    ("finished", 85, 90, "Training finished: compare test results"),
]


def read_demo(path=ROOT / "demo/demo-data.js"):
    raw = Path(path).read_text()
    return json.loads(raw.removeprefix("window.BBCSPORT_DEMO = ").removesuffix(";\n"))


def fingerprints():
    files = [ROOT / "demo/demo-data.js", Path(__file__), *sorted((ROOT / "colabeling").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def read_story(path=ROOT / "demo/story-data.js"):
    return json.loads(Path(path).read_text().removeprefix(PREFIX).removesuffix(";\n"))


def observe_weights(kernel, sheets, groups, config):
    """Record actual weight assignments in the original fit_mkl call.

    sys.settrace is an external observer. No callbacks, patches, altered
    tolerances or replacements are installed in the optimization code.
    The final polishing solution is recorded as a final solution, not as
    fabricated intermediate SLSQP iterations.
    """
    observed = []

    def observe(frame, event, arg):
        if frame.f_code is not fit_mkl.__code__:
            return None
        local = frame.f_locals
        if event == "line" and "inner" in local and "weights" in local:
            weights = local["weights"]
            if not observed or not np.array_equal(weights, observed[-1][1]):
                observed.append((int(local["inner"]), weights.copy()))
        return observe

    previous_trace = sys.gettrace()
    with threadpool_limits(limits=1):
        try:
            sys.settrace(observe)
            model = fit_mkl(
                kernel, sheets, groups, c=config["c"], grouped=True,
                max_inner=config["max_inner"], tolerance=config["tolerance"],
            )
        finally:
            sys.settrace(previous_trace)
    if not model.converged:
        raise RuntimeError("The focused replay did not converge; no animation was exported.")
    if not np.array_equal(observed[-1][1], model.weights):
        observed.append((model.inner_iterations, model.weights.copy()))
    snapshots = []
    for i, (iteration, weights) in enumerate(observed):
        label = "Initial weights" if i == 0 else f"Weight update {i}"
        if i == len(observed) - 1:
            label = "Final solution"
        snapshots.append({
            "label": label, "solver_iteration": iteration,
            "weights": weights.tolist(), "shares": (100 * weights / weights.sum()).tolist(),
        })
    # Sample observed assignments, retaining the actual final solution.
    indices = sorted({i for i in [0, 1, 2, 4, 8, 16, 32, 64, len(snapshots) - 2, len(snapshots) - 1]
                      if 0 <= i < len(snapshots)})
    return model, snapshots, indices


def build_story(*, refresh=False):
    source_hashes = fingerprints()
    destination = ROOT / "demo/story-data.js"
    if destination.exists() and not refresh:
        cached = read_story(destination)
        if cached.get("source_sha256") == source_hashes:
            return cached

    demo = read_demo()
    states = demo["methods"]["colabel_grouped"]["states"]
    if len(states) != 9:
        raise ValueError("This 90-second story requires the supplied eight-round recording.")
    training = sorted((a for a in demo["articles"] if a["role"] == "train"), key=lambda a: a["row"])
    unknown = [a for a in training if not a["known"]]
    rows = np.array([a["row"] for a in unknown])
    known = [a for a in training if a["known"]]
    if len(unknown) != 360 or len(known) != 10:
        raise ValueError("The presentation requires ten supplied labels and 360 unlabeled articles.")
    config = demo["experiment"]["config"]
    split = make_split(load_data(), demo["recording_seed"], config["labels_per_class"])
    if [int(i) for i in split.train] != [a["id"] for a in training]:
        raise ValueError("Local data split does not match the saved recording.")
    football = demo["categories"].index("football")
    detail = states[2]["details"]["categories"][football]["views"][0]
    entries = detail["sheets"]
    supplied = np.where(split.known_classes < 0, 0, np.where(split.known_classes == football, 1, -1))
    candidates = np.array([
        project_labels(
            np.array(states[e["origin_round"]]["train_views"])[e["source_view"], :, football],
            supplied, e["fraction"],
        ) for e in entries
    ])
    groups = np.array([e["bias"] for e in entries])
    kernel = split.train_views[0] @ split.train_views[0].T + 1
    model, snapshots, indices = observe_weights(kernel, candidates, groups, config)
    np.testing.assert_allclose(model.weights, [e["weight"] for e in entries], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        model.decision_function(kernel), np.array(states[2]["train_views"])[0, :, football],
        rtol=1e-6, atol=1e-7,
    )

    rounds = []
    for i, state in enumerate(states):
        predictions = np.array(state["train_predictions"])[rows]
        previous = np.array(states[max(i - 1, 0)]["train_predictions"])[rows]
        rounds.append({
            "round": i, "predictions": predictions.tolist(),
            "changed": (predictions != previous).tolist(),
            "changed_count": int((predictions != previous).sum()),
            "football_sheets": [] if i == 0 else [
                v["candidate_count"] for v in state["details"]["categories"][football]["views"]
            ],
        })
    source_scores = np.array(states[1]["train_views"])[1, rows, football]
    ranked = np.argsort(-source_scores, kind="stable")
    ranking = [dict(rank=i + 1, id=unknown[r]["id"] + 1, score=float(source_scores[r]), index=int(r))
               for i, r in enumerate(ranked)]
    # Cutoff boundary pairs explain differences without selecting by correctness.
    preview = [ranking[i - 1] for i in [36, 37, 72, 73, 108, 109]]
    sheets = []
    for e, labels in zip(entries, candidates):
        unknown_labels = labels[rows]
        sheets.append({
            "fraction": e["fraction"], "origin_round": e["origin_round"], "source_view": e["source_view"],
            "football_count": int((unknown_labels == 1).sum()), "labels": unknown_labels.tolist(),
            "preview": [int(unknown_labels[a["index"]]) for a in preview],
        })
    method_names = [
        ("supervised", "Ten labels only"), ("cotraining", "Co-training"),
        ("colabel_flat", "Co-labeling: 1 layer"), ("colabel_grouped", "Co-labeling: 2 layers"),
    ]
    results = [dict(method=key, label=label, **demo["experiment"]["methods"][key])
               for key, label in method_names]
    payload = {
        "schema_version": 1, "duration": 90,
        "chapters": [dict(id=k, start=a, end=b, title=t) for k, a, b, t in CHAPTERS],
        "source_sha256": source_hashes, "categories": demo["categories"], "colors": COLORS,
        "known": [dict(id=a["id"] + 1, category=a["answer"]) for a in known],
        "article_ids": [a["id"] + 1 for a in unknown], "rounds": rounds,
        "ranking": ranking, "preview": preview, "sheets": sheets,
        "optimizer": {"snapshots": snapshots, "display_indices": indices, "iterations": model.inner_iterations,
                      "observed_final_max_difference": float(np.abs(model.weights - np.array([e["weight"] for e in entries])).max())},
        "results": results, "evaluation_runs": len(demo["experiment"]["runs"]), "test_articles": 367,
        "notes": ["Predictions are saved outer-round states, not measurements between rounds.",
                  "Weight bars connect sampled real optimizer states; numerical labels are recorded values.",
                  "Influence is 100 * weight / sum(weights), not confidence or accuracy.",
                  "Final optimizer polishing is shown as the final solution, not as invented intermediate steps."],
    }
    destination.write_text(PREFIX + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n")
    return payload
