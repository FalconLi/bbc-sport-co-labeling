"""General multi-view co-labeling for SSL, MIL and relative outlier detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.svm import OneClassSVM, SVC

from .constraints import MILConstraints, RODConstraints, SSLConstraints
from .kernels import FeatureKernel, validate_views
from .solver import fit_mkl


@dataclass
class _Decision:
    coefficients: np.ndarray | None = None
    rho: float = 1.0
    initial_classifier: object | None = None

    def score(self, features, cross):
        if self.initial_classifier is not None:
            return self.initial_classifier.decision_function(features)
        if self.coefficients is None:
            raise ValueError("MIL bag-label initialization is not an out-of-sample classifier; use a trained round.")
        return cross @ self.coefficients / self.rho


def _mask(values, n, name):
    result = np.asarray(values)
    if result.shape != (n,) or not np.isin(result, [False, True]).all():
        raise ValueError(f"{name} must be a boolean vector with one entry per sample.")
    return result.astype(bool)


def _labels(values, n):
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype.kind not in "biufUS":
        raise ValueError("Labels must be a numeric or string vector with one entry per sample.")
    return result


class CoLabelingClassifier:
    """Paper Algorithms 1-3, supporting any V>=2 and one-vs-rest SSL classes.

    SSL: fit(views, y, known_mask=mask). Only y[mask] is read as training truth.
    MIL: fit(views, bag_labels_per_instance, bag_ids=ids). Labels describe bags.
    ROD: fit(views, known_normal_mask=mask). Only reference normals are supplied.

    layers=1 is flat MKL; layers=2 merges source/bias groups with L2; layers=3
    uses the paper's full source(L-infinity)/bias(L2)/iteration(L1) hierarchy.
    With two views, layers 2 and 3 have the same constraint.

    Kernel and normalization may be a common value or a list per view. RBF
    gamma=None fits inverse mean squared distance using training features only.
    ROD starts with OneClassSVM on reference normals (an explicit implementation
    choice because the paper does not specify its single-class initializer).
    """

    def __init__(self, *, task="ssl", layers=3, c=10.0, max_rounds=10, max_inner=100,
                 tolerance=1e-5, outer_tolerance=1e-6, kernel="linear", gamma=None,
                 normalization=None, proportion_offsets=None, mil_biases=(-.5, -.4, -.3),
                 minimum_positive=1, mil_positive_fraction=None, outlier_fraction=.1,
                 positive_label=None, normal_label="normal", outlier_label="outlier",
                 norms=None, record_history=False, class_priors=None):
        self.task, self.layers, self.c = task, layers, c
        self.max_rounds, self.max_inner = max_rounds, max_inner
        self.tolerance, self.outer_tolerance = tolerance, outer_tolerance
        self.kernel, self.gamma, self.normalization = kernel, gamma, normalization
        self.proportion_offsets = tuple(np.linspace(-.1, .1, 11)) if proportion_offsets is None else tuple(proportion_offsets)
        self.mil_biases = tuple(mil_biases)
        self.minimum_positive, self.mil_positive_fraction = minimum_positive, mil_positive_fraction
        self.outlier_fraction, self.positive_label = outlier_fraction, positive_label
        self.normal_label, self.outlier_label = normal_label, outlier_label
        self.norms, self.record_history = norms, record_history
        self.class_priors = None if class_priors is None else tuple(class_priors)

    def _parameters(self):
        return {name: getattr(self, name) for name in [
            "task", "layers", "c", "max_rounds", "max_inner", "tolerance", "outer_tolerance",
            "kernel", "gamma", "normalization", "proportion_offsets", "mil_biases",
            "minimum_positive", "mil_positive_fraction", "outlier_fraction", "positive_label",
            "normal_label", "outlier_label", "norms", "record_history", "class_priors",
        ]}

    def _validate_parameters(self):
        if self.task not in ("ssl", "mil", "rod") or self.layers not in (1, 2, 3):
            raise ValueError("task must be ssl/mil/rod and layers must be 1, 2 or 3.")
        for name in ["max_rounds", "max_inner"]:
            if not isinstance(getattr(self, name), (int, np.integer)) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ["c", "tolerance", "outer_tolerance"]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.norms is not None and self.layers == 1:
            raise ValueError("Custom hierarchical norms cannot be combined with layers=1.")
        if self.class_priors is not None and self.task != "ssl":
            raise ValueError("class_priors are available only for SSL.")

    def _make_constraints(self, y, known_mask, bag_ids, known_normal_mask):
        n = self.n_samples_
        if self.task == "rod":
            if y is not None or known_mask is not None or bag_ids is not None:
                raise ValueError("For ROD, supply only known_normal_mask; training outlier labels are not accepted.")
            if known_normal_mask is None:
                raise ValueError("ROD requires known_normal_mask.")
            self.classes_ = np.asarray([self.normal_label, self.outlier_label])
            if len(np.unique(self.classes_)) != 2:
                raise ValueError("Normal and outlier labels must differ.")
            return [RODConstraints(_mask(known_normal_mask, n, "known_normal_mask"), self.outlier_fraction, self.proportion_offsets)]
        if y is None or known_normal_mask is not None:
            raise ValueError("Supply SSL labels or per-instance MIL bag labels, not known_normal_mask.")
        y = _labels(y, n)
        mask = np.ones(n, dtype=bool) if known_mask is None else _mask(known_mask, n, "known_mask")
        if not mask.any():
            raise ValueError("At least one supplied label is required.")
        observed = y[mask]
        if observed.dtype.kind in "f" and not np.isfinite(observed).all():
            raise ValueError("Supplied labels must be finite.")
        self.classes_ = np.unique(observed)
        if len(self.classes_) < 2:
            raise ValueError("At least two observed classes are required for SSL/MIL.")
        if self.task == "mil":
            if known_mask is not None or bag_ids is None or len(self.classes_) != 2:
                raise ValueError("MIL needs exactly two bag-label classes and bag_ids; do not supply known_mask.")
            if self.positive_label is not None:
                if self.positive_label not in self.classes_:
                    raise ValueError("positive_label is not one of the bag-label classes.")
                self.classes_ = np.array([v for v in self.classes_ if v != self.positive_label] + [self.positive_label])
            binary = np.where(y == self.classes_[1], 1, -1)
            return [MILConstraints(bag_ids, binary, self.minimum_positive, self.mil_positive_fraction, self.mil_biases)]
        if bag_ids is not None:
            raise ValueError("bag_ids are used only for MIL.")
        priors = [None] * len(self.classes_)
        if self.class_priors is not None:
            priors = np.asarray(self.class_priors, dtype=float)
            if (priors.shape != (len(self.classes_),) or not np.isfinite(priors).all()
                    or np.any((priors < 0) | (priors > 1)) or not np.isclose(priors.sum(), 1)):
                raise ValueError("class_priors must be probabilities summing to one, in classes_ order.")
        result = []
        for label, prior in zip(self.classes_, priors):
            known = np.zeros(n, dtype=np.int8)
            known[mask] = np.where(y[mask] == label, 1, -1)
            result.append(SSLConstraints(known, self.proportion_offsets, prior))
        return result

    def fit(self, views, y=None, *, known_mask=None, bag_ids=None, known_normal_mask=None, progress=None):
        self._fitted = False
        self._validate_parameters()
        raw = validate_views(views)
        self.n_views_, self.n_samples_ = len(raw), raw[0].shape[0]
        self.feature_dimensions_ = [x.shape[1] for x in raw]
        def per_view(value):
            if isinstance(value, (list, tuple)):
                if len(value) != self.n_views_:
                    raise ValueError("Per-view kernel options must match the number of views.")
                return value
            return [value] * self.n_views_
        self.kernels_ = [FeatureKernel(k, g, norm) for k, g, norm in zip(
            per_view(self.kernel), per_view(self.gamma), per_view(self.normalization))]
        self.training_views_ = [spec.fit(x) for spec, x in zip(self.kernels_, raw)]
        constraints = self._make_constraints(y, known_mask, bag_ids, known_normal_mask)
        kernels = [spec.matrix(x, x) + 1.0 for spec, x in zip(self.kernels_, self.training_views_)]
        self._binary_runs = [self._fit_binary(constraint, kernels, index, len(constraints), progress)
                             for index, constraint in enumerate(constraints)]
        self.n_iter_ = [len(run["history"]) for run in self._binary_runs]
        self.stop_reasons_ = [run["stop_reason"] for run in self._binary_runs]
        self.outer_converged_ = all(reason != "round_limit" for reason in self.stop_reasons_)
        self._final_models = [[run["history"][-1]["models"][v] for run in self._binary_runs]
                              for v in range(self.n_views_)]
        self.kernel_weights_ = [run["final_sheets"] for run in self._binary_runs]
        self._collect_history()
        self._fitted = True
        return self

    def _fit_binary(self, constraint, kernels, category, total_categories, progress):
        initial_models, initial_scores = [], []
        for v, features in enumerate(self.training_views_):
            spec = self.kernels_[v]
            if self.task == "mil":
                decision, scores = _Decision(), constraint.initial_labels.astype(float)
            else:
                options = {"kernel": spec.kind}
                if spec.kind == "rbf":
                    options["gamma"] = spec.gamma
                if self.task == "rod":
                    classifier = OneClassSVM(nu=float(np.clip(self.outlier_fraction, .001, .999)), **options)
                    classifier.fit(features[constraint.known_normal])
                else:
                    classifier = SVC(C=self.c, **options)
                    mask = constraint.known != 0
                    classifier.fit(features[mask], constraint.known[mask])
                decision = _Decision(initial_classifier=classifier)
                scores = classifier.decision_function(features)
            initial_models.append(decision); initial_scores.append(scores)
        candidates = [[] for _ in kernels]
        metadata = [[] for _ in kernels]
        seen = [set() for _ in kernels]
        norms = self.norms if self.norms is not None else ((2, 2, 1) if self.layers == 2 else None)
        iteration_norm = 1 if norms is None else norms[2]
        def append(target, source_scores, origin_round):
            added = 0
            for source in range(self.n_views_):
                if source == target:
                    continue
                for bias, setting in enumerate(constraint.settings):
                    sheet = constraint.project(source_scores[source], float(setting))
                    key = (source, bias, sheet.tobytes())
                    if iteration_norm == 1 and key in seen[target]:
                        continue
                    seen[target].add(key)
                    candidates[target].append(sheet)
                    metadata[target].append(dict(bias=bias, origin_round=origin_round, source_view=source,
                                                 **constraint.metadata(setting)))
                    added += 1
            return added
        for target in range(self.n_views_):
            append(target, initial_scores, 0)
        unknown = np.flatnonzero(constraint.unknown_mask)
        preview = unknown[np.linspace(0, len(unknown) - 1, min(8, len(unknown)), dtype=int)] if len(unknown) else np.arange(min(8, self.n_samples_))
        history, previous, final_sheets = [], None, None
        stop_reason = "round_limit"
        for iteration in range(1, self.max_rounds + 1):
            scores, models, details, objectives, current_sheets = [], [], [], [], []
            for view in range(self.n_views_):
                sheets = np.asarray(candidates[view])
                groups = np.array([e["bias"] for e in metadata[view]], dtype=int)
                sources = np.array([e["source_view"] for e in metadata[view]], dtype=int)
                model = fit_mkl(kernels[view], sheets, groups, c=self.c, grouped=self.layers != 1,
                                max_inner=self.max_inner, tolerance=self.tolerance, source_groups=sources, norms=norms)
                if not model.converged:
                    raise RuntimeError(f"MKL did not converge for class {category}, view {view}, round {iteration} (relative gap {model.gap/max(-2*model.objective,1e-12):.3g}).")
                scores.append(model.decision_function(kernels[view]))
                models.append(_Decision(model.coefficients.copy(), model.rho))
                objectives.append(model.objective)
                total = model.weights.sum()
                weighted = [dict(e, weight=float(weight), share=float(weight / total))
                            for e, weight in zip(metadata[view], model.weights)]
                current_sheets.append(weighted)
                details.append({
                    "objective": model.objective, "gap": model.gap,
                    "relative_gap": model.gap / max(-2 * model.objective, 1e-12),
                    "inner_iterations": model.inner_iterations, "converged": model.converged,
                    "candidate_count": len(sheets),
                    "sheets": [dict(e, answers=sheets[i, preview].astype(int).tolist()) for i, e in enumerate(weighted)] if self.record_history else [],
                })
            history.append({"train": np.array(scores), "models": models, "views": details})
            final_sheets = current_sheets
            if progress:
                progress(f"class {category + 1}/{total_categories}, round {iteration}, sheets " + "/".join(str(len(c)) for c in candidates))
            added = sum(append(view, scores, iteration) for view in range(self.n_views_))
            stable = previous is not None and all(abs(now - old) <= self.outer_tolerance * max(abs(old), 1e-12)
                                                for now, old in zip(objectives, previous))
            if added == 0 or stable:
                stop_reason = "no_new_candidates" if added == 0 else "objective_stable"
                break
            previous = objectives
        return {"initial_models": initial_models, "initial_scores": np.array(initial_scores),
                "history": history, "preview": preview.tolist(), "stop_reason": stop_reason,
                "final_sheets": final_sheets}

    def _class_scores(self, binary):
        if self.task == "ssl":
            return binary
        f = binary[:, :, 0]
        return np.stack((f, -f) if self.task == "rod" else (-f, f), axis=2)

    def _collect_history(self):
        total = max(self.n_iter_)
        preview = self._binary_runs[0]["preview"]
        initial = np.stack([r["initial_scores"] for r in self._binary_runs], axis=2)
        self.train_scores_ = [self._class_scores(initial)]
        self.history_ = [{"round": 0, "preview_training_rows": preview, "categories": []}]
        if self.task != "ssl":
            self.history_[0]["initialization"] = "bag_labels" if self.task == "mil" else "reference_normal_one_class_svm"
        for iteration in range(total):
            selected = [r["history"][min(iteration, len(r["history"]) - 1)] for r in self._binary_runs]
            self.train_scores_.append(self._class_scores(np.stack([s["train"] for s in selected], axis=2)))
            categories = [dict(category=i, views=state["views"], completed_rounds=self.n_iter_[i]) for i, state in enumerate(selected)]
            self.history_.append({"round": iteration + 1, "preview_training_rows": preview, "categories": categories})

    def _prepare_prediction(self, views):
        if not getattr(self, "_fitted", False):
            raise ValueError("Fit or load the model before predicting.")
        inputs = validate_views(views, self.feature_dimensions_)
        prepared = [spec.prepare(x) for spec, x in zip(self.kernels_, inputs)]
        cross = [spec.matrix(x, training) + 1.0 for spec, x, training in zip(self.kernels_, prepared, self.training_views_)]
        return prepared, cross

    def _score_prepared(self, features, cross, round_index=None):
        if round_index is None:
            models = self._final_models
        else:
            if (not isinstance(round_index, (int, np.integer)) or not hasattr(self, "_binary_runs")
                    or not 0 <= round_index <= max(self.n_iter_)):
                raise ValueError("That round is unavailable; model archives store final classifiers only.")
            models = [[(run["initial_models"][v] if round_index == 0 else
                        run["history"][min(round_index - 1, len(run["history"]) - 1)]["models"][v])
                       for run in self._binary_runs] for v in range(self.n_views_)]
        binary = np.array([np.column_stack([m.score(features[v], cross[v]) for m in models[v]]) for v in range(self.n_views_)])
        return self._class_scores(binary)

    def decision_function_views(self, views, *, round_index=None):
        """Per-view scores [view, sample, class]; higher is stronger for a class."""
        features, cross = self._prepare_prediction(views)
        return self._score_prepared(features, cross, round_index)

    def decision_function(self, views):
        return self.decision_function_views(views).mean(axis=0)

    def predict(self, views):
        return self.classes_[self.decision_function(views).argmax(axis=1)]

    def score_history(self, views):
        """Prediction replay; MIL starts at round 1 because round 0 is bag labels."""
        features, cross = self._prepare_prediction(views)
        if not hasattr(self, "_binary_runs"):
            raise ValueError("Round history is not stored in the model archive.")
        first = 1 if self.task == "mil" else 0
        return [self._score_prepared(features, cross, i) for i in range(first, max(self.n_iter_) + 1)]

    def bag_decision_function(self, views, bag_ids):
        if self.task != "mil":
            raise ValueError("Bag predictions are available only for MIL models.")
        scores = self.decision_function(views)[:, 1]
        ids = np.asarray(bag_ids)
        if ids.shape != scores.shape:
            raise ValueError("bag_ids must have one entry per prediction instance.")
        bags = np.unique(ids)
        maximum = np.array([scores[ids == bag].max() for bag in bags])
        return bags, np.column_stack((-maximum, maximum))

    def predict_bags(self, views, bag_ids):
        bags, scores = self.bag_decision_function(views, bag_ids)
        return bags, self.classes_[scores.argmax(axis=1)]

    def save(self, path):
        """Save a portable NPZ of numeric/string arrays; no pickle or code loading."""
        if not getattr(self, "_fitted", False):
            raise ValueError("Fit the model before saving.")
        params = self._parameters()
        if params["norms"] is not None:
            params["norms"] = ["inf" if np.isinf(p) else p for p in params["norms"]]
        metadata = {"format_version": 1, "parameters": params, "classes": self.classes_.tolist(),
                    "class_dtype": str(self.classes_.dtype), "dimensions": self.feature_dimensions_,
                    "n_iter": self.n_iter_, "stop_reasons": self.stop_reasons_, "kernels": [], "view_formats": [],
                    "kernel_weights": self.kernel_weights_}
        arrays = {}
        for v, (spec, training) in enumerate(zip(self.kernels_, self.training_views_)):
            metadata["kernels"].append({"kind": spec.kind, "gamma": spec.gamma, "normalization": spec.normalization})
            if sparse.issparse(training):
                training = sparse.csr_matrix(training)
                metadata["view_formats"].append("csr")
                for name in ["data", "indices", "indptr"]:
                    arrays[f"view_{v}_{name}"] = getattr(training, name)
                arrays[f"view_{v}_shape"] = np.array(training.shape)
            else:
                metadata["view_formats"].append("dense")
                arrays[f"view_{v}"] = training
            arrays[f"coefficients_{v}"] = np.column_stack([m.coefficients for m in self._final_models[v]])
            arrays[f"rho_{v}"] = np.array([m.rho for m in self._final_models[v]])
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False,
                                              default=lambda value: value.item() if isinstance(value, np.generic) else list(value)))
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format_version") != 1:
                raise ValueError("Unsupported model archive version.")
            params = metadata["parameters"]
            if params.get("norms") is not None:
                params["norms"] = tuple(float(p) for p in params["norms"])
            model = cls(**params)
            model._validate_parameters()
            model.classes_ = np.asarray(metadata["classes"], dtype=metadata["class_dtype"])
            if (model.classes_.ndim != 1 or len(model.classes_) < 2 or model.classes_.dtype.kind not in "biufUS"
                    or len(np.unique(model.classes_)) != len(model.classes_)):
                raise ValueError("Invalid stored class labels.")
            if model.classes_.dtype.kind == "f" and not np.isfinite(model.classes_).all():
                raise ValueError("Stored class labels must be finite.")
            if model.task != "ssl" and len(model.classes_) != 2:
                raise ValueError("MIL/ROD archives require two class labels.")
            model.feature_dimensions_ = metadata["dimensions"]
            model.n_views_ = len(model.feature_dimensions_)
            model.kernels_ = [FeatureKernel(**spec) for spec in metadata["kernels"]]
            if len(model.kernels_) != model.n_views_ or len(metadata["view_formats"]) != model.n_views_:
                raise ValueError("Model view metadata is inconsistent.")
            if any(spec.kind == "rbf" and spec.gamma is None for spec in model.kernels_):
                raise ValueError("Stored RBF kernels require their fitted gamma.")
            model.training_views_, model._final_models = [], []
            for v, kind in enumerate(metadata["view_formats"]):
                if kind == "csr":
                    x = sparse.csr_matrix((archive[f"view_{v}_data"], archive[f"view_{v}_indices"], archive[f"view_{v}_indptr"]),
                                          shape=tuple(archive[f"view_{v}_shape"]))
                    x.check_format(full_check=True)
                elif kind == "dense":
                    x = archive[f"view_{v}"].copy()
                else:
                    raise ValueError("Invalid stored feature format.")
                coefficients = np.asarray(archive[f"coefficients_{v}"], dtype=float)
                rho = np.asarray(archive[f"rho_{v}"], dtype=float)
                count = len(model.classes_) if model.task == "ssl" else 1
                if (coefficients.shape != (x.shape[0], count) or rho.shape != (count,)
                        or not np.isfinite(coefficients).all() or not np.isfinite(rho).all() or np.any(rho <= 0)):
                    raise ValueError("Invalid model coefficients or normalization.")
                model.training_views_.append(x)
                model._final_models.append([_Decision(coefficients[:, k].copy(), float(rho[k])) for k in range(count)])
        validate_views(model.training_views_, model.feature_dimensions_)
        model.n_samples_ = model.training_views_[0].shape[0]
        model.n_iter_, model.stop_reasons_ = metadata["n_iter"], metadata["stop_reasons"]
        model.outer_converged_ = all(reason != "round_limit" for reason in model.stop_reasons_)
        model.history_, model.train_scores_, model.kernel_weights_ = [], [], metadata.get("kernel_weights", [])
        model._fitted = True
        return model
