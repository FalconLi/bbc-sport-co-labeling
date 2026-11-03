"""The SSL, MIL and ROD feasible label sets from Sections 3.2 and 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import project_labels


def _vector(values, name, length=None):
    result = np.asarray(values)
    if result.ndim != 1 or not len(result) or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be a nonempty vector with one entry per sample.")
    return result


@dataclass
class SSLConstraints:
    known: np.ndarray
    offsets: tuple = tuple(np.linspace(-0.1, 0.1, 11))
    positive_fraction: float | None = None

    def __post_init__(self):
        self.known = _vector(self.known, "known labels")
        if not np.isin(self.known, [-1, 0, 1]).all() or not np.any(self.known):
            raise ValueError("SSL needs supplied +1/-1 labels; use 0 only for unknown labels.")
        self.known = self.known.astype(np.int8)
        fraction = float((self.known[self.known != 0] == 1).mean()) if self.positive_fraction is None else self.positive_fraction
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("The positive fraction must lie in [0, 1].")
        offsets = np.asarray(self.offsets, dtype=float)
        if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all():
            raise ValueError("At least one finite proportion offset is required.")
        self.settings = np.clip(fraction + offsets, 0, 1)

    @property
    def unknown_mask(self):
        return self.known == 0

    def project(self, scores, setting):
        return project_labels(scores, self.known, float(setting))

    def metadata(self, setting):
        return {"fraction": round(float(setting), 6)}


@dataclass
class MILConstraints:
    bag_ids: np.ndarray
    bag_labels: np.ndarray
    minimum_positive: int = 1
    positive_fraction: float | None = None
    biases: tuple = (-0.5, -0.4, -0.3)

    def __post_init__(self):
        self.bag_ids = _vector(self.bag_ids, "bag_ids")
        if self.bag_ids.dtype.kind not in "biufUS" or (self.bag_ids.dtype.kind in "f" and not np.isfinite(self.bag_ids).all()):
            raise ValueError("Bag identifiers must be finite numeric or string values.")
        labels = _vector(self.bag_labels, "bag_labels", len(self.bag_ids))
        if not np.isin(labels, [-1, 1]).all():
            raise ValueError("Internal bag labels must be +1/-1, repeated for each bag's instances.")
        if (not isinstance(self.minimum_positive, (int, np.integer)) or self.minimum_positive < 1):
            raise ValueError("minimum_positive must be a positive integer.")
        if self.positive_fraction is not None and (not np.isfinite(self.positive_fraction) or not 0 < self.positive_fraction <= 1):
            raise ValueError("The MIL positive fraction must lie in (0, 1].")
        self.groups = [np.flatnonzero(self.bag_ids == bag) for bag in np.unique(self.bag_ids)]
        self.required = []
        for indices in self.groups:
            if len(np.unique(labels[indices])) != 1:
                raise ValueError("All instances in a bag must carry the same bag label.")
            count = (int(np.ceil(len(indices) * self.positive_fraction)) if self.positive_fraction is not None
                     else self.minimum_positive)
            if labels[indices[0]] == 1 and count > len(indices):
                raise ValueError("A positive bag has fewer instances than its required positive count.")
            self.required.append(count)
        self.bag_labels = labels.astype(np.int8)
        self.settings = np.asarray(self.biases, dtype=float)
        if self.settings.ndim != 1 or not len(self.settings) or not np.isfinite(self.settings).all():
            raise ValueError("At least one finite MIL prediction bias is required.")
        self.initial_labels = self.bag_labels.copy()

    @property
    def unknown_mask(self):
        return self.bag_labels == 1

    def project(self, scores, setting):
        scores = np.asarray(scores, dtype=float)
        if scores.shape != self.bag_ids.shape or not np.isfinite(scores).all() or not np.isfinite(setting):
            raise ValueError("MIL scores must be finite, with one value per instance.")
        adjusted = scores + setting
        result = np.full(len(scores), -1, dtype=np.int8)
        for indices, required in zip(self.groups, self.required):
            if self.bag_labels[indices[0]] == -1:
                continue
            result[indices[adjusted[indices] >= 0]] = 1
            if np.count_nonzero(result[indices] == 1) < required:
                ranked = indices[np.argsort(-adjusted[indices], kind="stable")]
                result[ranked[:required]] = 1
        return result

    def metadata(self, setting):
        return {"prediction_bias": float(setting)}


@dataclass
class RODConstraints:
    known_normal: np.ndarray
    outlier_fraction: float
    offsets: tuple = tuple(np.linspace(-0.1, 0.1, 11))

    def __post_init__(self):
        mask = _vector(self.known_normal, "known_normal_mask")
        if not np.isin(mask, [False, True]).all() or not mask.any() or mask.all():
            raise ValueError("ROD needs at least one reference normal and one unlabeled sample.")
        self.known_normal = mask.astype(bool)
        if not np.isfinite(self.outlier_fraction) or not 0 < self.outlier_fraction < 1:
            raise ValueError("ROD requires an outlier fraction strictly between zero and one.")
        offsets = np.asarray(self.offsets, dtype=float)
        if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all():
            raise ValueError("At least one finite outlier-ratio offset is required.")
        self.settings = np.clip(self.outlier_fraction + offsets, 0, 1)
        # The paper assigns +1 to normal samples and -1 to outliers.
        self.known = self.known_normal.astype(np.int8)

    @property
    def unknown_mask(self):
        return ~self.known_normal

    def project(self, scores, setting):
        return project_labels(scores, self.known, 1 - float(setting))

    def metadata(self, setting):
        return {"outlier_fraction": round(float(setting), 6)}
