"""Experiment adapters and conventional baselines over arbitrary view/class counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.svm import SVC

from .estimator import CoLabelingClassifier
from .kernels import validate_views


@dataclass
class LearningRun:
    # One entry per round; scores have shape [view, sample, category].
    train_scores: list[np.ndarray]
    test_scores: list[np.ndarray]
    details: list[dict]
    model: CoLabelingClassifier | None = None


def _inputs(train_views, test_views, supplied):
    train_views = validate_views(train_views)
    test_views = validate_views(test_views, [x.shape[1] for x in train_views])
    supplied = np.asarray(supplied)
    if supplied.shape != (train_views[0].shape[0],) or supplied.dtype.kind not in "iu":
        raise ValueError("The experiment adapter needs integer labels and -1 for unknown labels.")
    classes = np.unique(supplied[supplied >= 0])
    if len(classes) < 2 or np.any(supplied < -1):
        raise ValueError("At least two known classes are needed; only -1 denotes an unknown label.")
    return train_views, test_views, supplied, classes


def _supervised_views(train_views, test_views, supplied, c):
    train_views, test_views, supplied, classes = _inputs(train_views, test_views, supplied)
    mask = supplied >= 0
    train_scores = np.empty((len(train_views), len(supplied), len(classes)))
    test_scores = np.empty((len(train_views), test_views[0].shape[0], len(classes)))
    for view in range(len(train_views)):
        for category, label in enumerate(classes):
            classifier = SVC(C=c, kernel="linear")
            answers = np.where(supplied[mask] == label, 1, -1)
            classifier.fit(train_views[view][mask], answers)
            train_scores[view, :, category] = classifier.decision_function(train_views[view])
            test_scores[view, :, category] = classifier.decision_function(test_views[view])
    return train_scores, test_scores


def fit_supervised(train_views, test_views, supplied, *, c=10.0) -> LearningRun:
    train, test = _supervised_views(train_views, test_views, supplied, c)
    return LearningRun([train], [test], [{"round": 0, "known_answers": int((np.asarray(supplied) >= 0).sum())}])


def fit_cotraining(train_views, test_views, supplied, *, c=10.0, rounds=10, batch=5) -> LearningRun:
    """Confident-sample exchange baseline; other views' scores are averaged.

    Each teacher set supplies the highest/lowest batch still-unassigned examples
    per binary class. With two views this is exactly the original BBCSport
    selection policy, which the paper does not specify in full.
    """
    train_views, test_views, supplied, classes = _inputs(train_views, test_views, supplied)
    if rounds < 1 or batch < 1:
        raise ValueError("rounds and batch must be positive.")
    n, nt, nv, nc = len(supplied), test_views[0].shape[0], len(train_views), len(classes)
    assigned = np.zeros((nv, nc, n), dtype=np.int8)
    known = supplied >= 0
    for category, label in enumerate(classes):
        assigned[:, category, known] = np.where(supplied[known] == label, 1, -1)
    train_history, test_history, details = [], [], []
    for iteration in range(rounds + 1):
        train_scores, test_scores = np.empty((nv, n, nc)), np.empty((nv, nt, nc))
        for view in range(nv):
            for category in range(nc):
                use = assigned[view, category] != 0
                model = SVC(C=c, kernel="linear")
                model.fit(train_views[view][use], assigned[view, category, use])
                train_scores[view, :, category] = model.decision_function(train_views[view])
                test_scores[view, :, category] = model.decision_function(test_views[view])
        train_history.append(train_scores)
        test_history.append(test_scores)
        details.append({"round": iteration, "assigned_per_view_category": (assigned != 0).sum(axis=2).tolist()})
        if iteration == rounds:
            break
        updated = assigned.copy()
        for student in range(nv):
            teachers = [v for v in range(nv) if v != student]
            for category in range(nc):
                unknown = np.flatnonzero(assigned[student, category] == 0)
                scores = train_scores[teachers, :, category].mean(axis=0)
                ranking = unknown[np.argsort(-scores[unknown], kind="stable")]
                count = min(batch, len(ranking) // 2)
                if count:
                    updated[student, category, ranking[:count]] = 1
                    updated[student, category, ranking[-count:]] = -1
        assigned = updated
    return LearningRun(train_history, test_history, details)


def fit_colabeling(
    train_views, test_views, supplied, *, c=10.0, rounds=10,
    grouped=True, max_inner=100, tolerance=1e-5,
    progress: Callable[[str], None] | None = None,
) -> LearningRun:
    """BBCSport-compatible adapter around the general estimator.

    The reusable CoLabelingClassifier also accepts string labels, known-label
    masks, any number of views/classes, MIL bags and ROD reference normals.
    This adapter retains the original experiment's -1 unknown-label convention
    and saved replay format.
    """
    train_views, test_views, supplied, _ = _inputs(train_views, test_views, supplied)
    model = CoLabelingClassifier(layers=3 if grouped else 1, c=c, max_rounds=rounds,
                                max_inner=max_inner, tolerance=tolerance, record_history=True)
    model.fit(train_views, supplied, known_mask=supplied >= 0, progress=progress)
    return LearningRun(model.train_scores_, model.score_history(test_views), model.history_, model)
