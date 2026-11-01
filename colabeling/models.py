"""Two-view classifiers. Training interfaces never receive hidden/test answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.svm import SVC

from .solver import fit_mkl, project_labels


@dataclass
class LearningRun:
    # One entry per round; scores have shape [view, article, category].
    train_scores: list[np.ndarray]
    test_scores: list[np.ndarray]
    details: list[dict]


def _supervised_views(train_views, test_views, supplied, c):
    mask = supplied >= 0
    train_scores = np.empty((2, len(supplied), 5))
    test_scores = np.empty((2, len(test_views[0]), 5))
    for view in range(2):
        for category in range(5):
            classifier = SVC(C=c, kernel="linear")
            answers = np.where(supplied[mask] == category, 1, -1)
            classifier.fit(train_views[view][mask], answers)
            train_scores[view, :, category] = classifier.decision_function(train_views[view])
            test_scores[view, :, category] = classifier.decision_function(test_views[view])
    return train_scores, test_scores


def fit_supervised(train_views, test_views, supplied, *, c=10.0) -> LearningRun:
    train, test = _supervised_views(train_views, test_views, supplied, c)
    return LearningRun([train], [test], [{"round": 0, "known_answers": int((supplied >= 0).sum())}])


def fit_cotraining(train_views, test_views, supplied, *, c=10.0, rounds=10, batch=5) -> LearningRun:
    """Classical confident-sample exchange baseline, five per binary class/view.

    At each round, teacher ranks its student's still-unassigned articles and
    supplies the highest/lowest five as positive/negative. Labels accumulate;
    the original ten human answers are never changed. This selection policy
    is explicit because the journal paper does not fully specify its baseline.
    """
    n, nt = len(supplied), len(test_views[0])
    assigned = np.zeros((2, 5, n), dtype=np.int8)
    known = supplied >= 0
    for category in range(5):
        assigned[:, category, known] = np.where(supplied[known] == category, 1, -1)
    train_history, test_history, details = [], [], []
    for iteration in range(rounds + 1):
        train_scores, test_scores = np.empty((2, n, 5)), np.empty((2, nt, 5))
        for view in range(2):
            for category in range(5):
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
        for student in range(2):
            for category in range(5):
                unknown = np.flatnonzero(assigned[student, category] == 0)
                ranking = unknown[np.argsort(-train_scores[1 - student, unknown, category], kind="stable")]
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
    """Algorithms 1-3 of Xu et al. specialized to the two-view SSL setting.

    No hidden labels are accepted here: supplied has category IDs for known
    examples and -1 for all unknown examples. The test inputs are features
    only, used exclusively to record predictions after fitting each model.
    """
    n, nt = len(supplied), len(test_views[0])
    known_mask = supplied >= 0
    initial_train, initial_test = _supervised_views(train_views, test_views, supplied, c)
    kernels = [x @ x.T + 1.0 for x in train_views]
    cross = [test_views[v] @ train_views[v].T + 1.0 for v in range(2)]
    offsets = np.linspace(-0.1, 0.1, 11)
    unknown_rows = np.flatnonzero(~known_mask)
    preview_rows = unknown_rows[np.linspace(0, len(unknown_rows) - 1, 8, dtype=int)]
    by_category = []
    for category in range(5):
        supplied_binary = np.zeros(n, dtype=np.int8)
        supplied_binary[known_mask] = np.where(supplied[known_mask] == category, 1, -1)
        prior = float((supplied[known_mask] == category).mean())
        fractions = np.clip(prior + offsets, 0.0, 1.0)
        candidates = [[], []]
        metadata = [[], []]
        seen = [set(), set()]

        def append_sheets(target_view, teacher_scores, origin_round):
            added = 0
            for bias, fraction in enumerate(fractions):
                sheet = project_labels(teacher_scores, supplied_binary, float(fraction))
                # Identical kernels within the same L1 history group can be
                # merged exactly; duplicate kernels in different bias groups
                # stay separate because their regularization is different.
                key = (bias, sheet.tobytes())
                if key in seen[target_view]:
                    continue
                seen[target_view].add(key)
                candidates[target_view].append(sheet)
                metadata[target_view].append({
                    "bias": bias, "fraction": round(float(fraction), 6),
                    "origin_round": origin_round, "source_view": 1 - target_view,
                })
                added += 1
            return added

        for view in range(2):
            append_sheets(view, initial_train[1 - view, :, category], 0)
        history = []
        previous_objectives = None
        for iteration in range(1, rounds + 1):
            trains, tests, view_details = [], [], []
            objectives = []
            for view in range(2):
                sheets = np.asarray(candidates[view])
                groups = np.array([entry["bias"] for entry in metadata[view]])
                model = fit_mkl(
                    kernels[view], sheets, groups, c=c, grouped=grouped,
                    max_inner=max_inner, tolerance=tolerance,
                )
                trains.append(model.decision_function(kernels[view]))
                tests.append(model.decision_function(cross[view]))
                objectives.append(model.objective)
                total = model.weights.sum()
                view_details.append({
                    "objective": model.objective, "gap": model.gap,
                    "relative_gap": model.gap / max(-2 * model.objective, 1e-12),
                    "inner_iterations": model.inner_iterations, "converged": model.converged,
                    "candidate_count": len(sheets),
                    "sheets": [dict(
                        entry, weight=float(weight), share=float(weight / total),
                        answers=sheets[index, preview_rows].astype(int).tolist(),
                    ) for index, (entry, weight) in enumerate(zip(metadata[view], model.weights))],
                })
            history.append({"train": np.asarray(trains), "test": np.asarray(tests), "views": view_details})
            if progress:
                progress(f"class {category + 1}/5, round {iteration}, "
                         f"sheets {len(candidates[0])}/{len(candidates[1])}")
            added = sum(append_sheets(view, trains[1 - view], iteration) for view in range(2))
            stable = previous_objectives is not None and all(
                abs(now - old) <= 1e-6 * max(abs(old), 1e-12)
                for now, old in zip(objectives, previous_objectives)
            )
            if added == 0 or stable:
                break
            previous_objectives = objectives
        by_category.append(history)

    total_rounds = max(len(history) for history in by_category)
    train_history, test_history = [initial_train], [initial_test]
    details = [{"round": 0, "preview_training_rows": preview_rows.tolist(), "categories": []}]
    for iteration in range(total_rounds):
        train_scores, test_scores = np.empty((2, n, 5)), np.empty((2, nt, 5))
        categories = []
        for category, history in enumerate(by_category):
            state = history[min(iteration, len(history) - 1)]
            train_scores[:, :, category], test_scores[:, :, category] = state["train"], state["test"]
            categories.append({"category": category, "views": state["views"], "completed_rounds": len(history)})
        train_history.append(train_scores)
        test_history.append(test_scores)
        details.append({"round": iteration + 1, "preview_training_rows": preview_rows.tolist(), "categories": categories})
    return LearningRun(train_history, test_history, details)
