"""Export real predictions, alternatives, and derived word clues for the demo.

The committed recording contains no full article text. The local demo server
can show original headlines/excerpts from the ignored downloaded source data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import CATEGORIES, ROOT
from .experiment import METHODS, METHOD_NAMES


def export_demo(dataset, report, recording, destination=ROOT / "demo" / "demo-data.js"):
    words = [np.array(indices) for indices in recording["word_indices"]]
    known = dict(zip(recording["train_indices"], recording["known_mask"]))
    articles = []
    for role in ["train", "test"]:
        for row, index in enumerate(recording[f"{role}_indices"]):
            clues = []
            for indices in words:
                counts = dataset.counts[index, indices].toarray().ravel()
                top = np.argsort(-counts, kind="stable")[:8]
                clues.append([dataset.terms[int(indices[t])] for t in top if counts[t] > 0])
            articles.append({
                "id": int(index), "role": role, "row": row,
                "known": bool(known.get(index, False)),
                "answer": int(dataset.labels[index]), "words": clues,
            })
    methods = {}
    for method in METHODS:
        learned = recording["methods"][method]
        states = []
        for round_index, (train, test) in enumerate(zip(learned["train_scores"], learned["test_scores"])):
            train, test = np.array(train), np.array(test)
            states.append({
                "round": round_index,
                "train_views": np.round(train, 8).tolist(), "test_views": np.round(test, 8).tolist(),
                "train_predictions": train.mean(axis=0).argmax(axis=1).tolist(),
                "test_predictions": test.mean(axis=0).argmax(axis=1).tolist(),
                "metrics": learned["round_metrics"][round_index],
                "details": learned["details"][round_index],
            })
        methods[method] = {"name": METHOD_NAMES[method], "states": states}
    payload = {
        "schema_version": 1, "recording_seed": recording["seed"],
        "categories": CATEGORIES, "articles": articles, "methods": methods,
        "experiment": report,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("window.BBCSPORT_DEMO = " + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n")
    print(f"Saved the presentation recording to {destination}", flush=True)
