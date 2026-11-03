"""Evaluation of saved classifiers; no evaluation labels enter training."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score


def _ap(positive, scores):
    # Match sklearn's zero convention without warnings for absent test classes.
    return float(100 * average_precision_score(positive, scores)) if np.any(positive) else 0.0


def classification_metrics(labels, view_scores, classes=None):
    views = np.asarray(view_scores, dtype=float)
    if views.ndim != 3 or not views.shape[0] or not views.shape[1] or not np.isfinite(views).all():
        raise ValueError("Scores must be finite [view, sample, class] arrays.")
    labels = np.asarray(labels)
    classes = np.arange(views.shape[2]) if classes is None else np.asarray(classes)
    if labels.shape != (views.shape[1],) or classes.shape != (views.shape[2],):
        raise ValueError("Label or class dimensions do not match the prediction scores.")
    if not np.isin(labels, classes).all():
        raise ValueError("Evaluation labels include classes absent from the fitted model.")
    scores = views.mean(axis=0)
    per_class = [_ap(labels == label, scores[:, i]) for i, label in enumerate(classes)]
    return {
        "map": float(np.mean(per_class)), "ap_by_class": per_class,
        "accuracy": float(100 * accuracy_score(labels, classes[scores.argmax(axis=1)])),
        "map_by_view": [float(np.mean([_ap(labels == label, view[:, i]) for i, label in enumerate(classes)])) for view in views],
    }


def evaluate_model(model, views, labels, *, bag_ids=None):
    labels = np.asarray(labels)
    if model.task == "ssl":
        if bag_ids is not None:
            raise ValueError("Bag evaluation is available only for MIL.")
        metrics = classification_metrics(labels, model.decision_function_views(views), model.classes_)
        metrics.update(task=model.task, unit="instance", samples=len(labels), classes=model.classes_.tolist())
        return metrics
    if bag_ids is not None:
        if model.task != "mil":
            raise ValueError("Bag evaluation is available only for MIL.")
        ids = np.asarray(bag_ids)
        if labels.shape != ids.shape:
            raise ValueError("Bag evaluation labels must be repeated per instance.")
        bags, scores = model.bag_decision_function(views, ids)
        truth = []
        for bag in bags:
            values = np.unique(labels[ids == bag])
            if len(values) != 1:
                raise ValueError("Test instances in a bag must have one consistent bag label.")
            truth.append(values[0])
        labels = np.asarray(truth)
        unit = "bag"
    else:
        scores = model.decision_function(views)
        unit = "instance"
    if labels.shape != (scores.shape[0],) or not np.isin(labels, model.classes_).all():
        raise ValueError("Test labels must match the model's normal/outlier or negative/positive classes.")
    positive = labels == model.classes_[1]
    return {"task": model.task, "unit": unit, "samples": len(labels), "classes": model.classes_.tolist(),
            "positive_class": model.classes_[1].item(), "ap": _ap(positive, scores[:, 1]),
            "accuracy": float(100 * accuracy_score(labels, model.classes_[scores.argmax(axis=1)]))}
