"""Validated view inputs and train-fitted linear/RBF kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.preprocessing import normalize


def validate_views(views, dimensions=None):
    if not isinstance(views, (list, tuple)) or len(views) < 2:
        raise ValueError("Provide at least two views as a list/tuple of sample-by-feature matrices.")
    if dimensions is not None and len(views) != len(dimensions):
        raise ValueError("The number of views differs from the fitted model.")
    result, samples = [], None
    for index, view in enumerate(views):
        if sparse.issparse(view):
            x = sparse.csr_matrix(view, dtype=np.float64)
            finite = np.isfinite(x.data).all()
        else:
            x = np.asarray(view, dtype=np.float64)
            finite = np.isfinite(x).all()
        if x.ndim != 2 or min(x.shape) < 1 or not finite:
            raise ValueError(f"View {index} must be a nonempty finite numeric matrix.")
        if samples is not None and x.shape[0] != samples:
            raise ValueError("Views must contain the same samples in the same row order.")
        if dimensions is not None and x.shape[1] != dimensions[index]:
            raise ValueError(f"View {index} has the wrong feature dimension.")
        samples = x.shape[0]
        result.append(x)
    return result


@dataclass
class FeatureKernel:
    kind: str = "linear"
    gamma: float | None = None
    normalization: str | None = None

    def __post_init__(self):
        if self.kind not in ("linear", "rbf"):
            raise ValueError("Kernels must be 'linear' or 'rbf'.")
        if self.normalization not in (None, "l1", "l2"):
            raise ValueError("Normalization must be None, 'l1', or 'l2'.")
        if self.gamma is not None and (not np.isfinite(self.gamma) or self.gamma <= 0):
            raise ValueError("RBF gamma must be finite and positive.")

    def prepare(self, x):
        return normalize(x, norm=self.normalization) if self.normalization is not None else x

    def fit(self, x):
        x = self.prepare(x)
        if self.kind == "rbf" and self.gamma is None:
            mean_squared_distance = float(euclidean_distances(x, squared=True).mean())
            self.gamma = 1 / mean_squared_distance if mean_squared_distance > 0 else 1.0
        return x

    def matrix(self, x, training):
        if self.kind == "rbf":
            return rbf_kernel(x, training, gamma=self.gamma)
        result = x @ training.T
        return result.toarray() if sparse.issparse(result) else np.asarray(result)
