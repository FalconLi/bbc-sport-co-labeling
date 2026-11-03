"""Source-view / bias / iteration kernel regularization (paper, Eqs. 11-17)."""

from __future__ import annotations

import numpy as np


def _norm(x, p):
    return float(np.max(x, initial=0)) if np.isinf(p) else float(np.linalg.norm(x, ord=p))


def _dual(p):
    return 1.0 if np.isinf(p) else (np.inf if p == 1 else p / (p - 1))


def _derivative(x, p):
    if p == 1:
        return np.ones_like(x)
    if np.isinf(p):
        selected = x == np.max(x, initial=0)
        return selected.astype(float) / max(1, selected.sum())
    value = _norm(x, p)
    return (x / value) ** (p - 1) if value > 0 else np.zeros_like(x)


class KernelHierarchy:
    """Three nested norms, in source / bias / iteration order.

    The paper writes the subscripts in the reverse order. Its recommended
    ell_(1,2,infinity) model therefore uses norms=(infinity,2,1) here.
    norms=(2,2,1) is its two-layer ablation, merging source and bias levels.
    Every leaf represents a candidate from an iteration. Ragged histories and
    missing bias groups are allowed; source identities never get merged.
    """

    def __init__(self, sources, biases, norms=(np.inf, 2.0, 1.0)):
        sources, biases = np.asarray(sources), np.asarray(biases)
        if sources.ndim != 1 or sources.shape != biases.shape or not len(sources):
            raise ValueError("Source and bias indices must be matching, nonempty vectors.")
        if (not np.issubdtype(sources.dtype, np.integer) or not np.issubdtype(biases.dtype, np.integer)
                or np.any(sources < 0) or np.any(biases < 0)):
            raise ValueError("Source and bias indices must be nonnegative integers.")
        self.norms = tuple(float(p) for p in norms)
        if len(self.norms) != 3 or any(np.isnan(p) or p < 1 for p in self.norms):
            raise ValueError("Three norm exponents >= 1 (or infinity) are required.")
        self.sources = np.unique(sources)
        self.groups = [
            [np.flatnonzero((sources == source) & (biases == bias))
             for bias in np.unique(biases[sources == source])]
            for source in self.sources
        ]
        self.n = len(sources)

    def _levels(self, values, norms=None):
        p1, p2, p3 = self.norms if norms is None else norms
        biases = [np.array([_norm(values[g], p3) for g in groups]) for groups in self.groups]
        sources = np.array([_norm(b, p2) for b in biases])
        return biases, sources, _norm(sources, p1)

    def norm(self, weights):
        return self._levels(np.asarray(weights))[2]

    def support(self, energies):
        """Exact support function of the nested-norm unit ball (dual norm)."""
        return self._levels(np.asarray(energies), tuple(_dual(p) for p in self.norms))[2]

    def update(self, squared_norms):
        """Recursive closed-form minimizer of sum_m squared_norms[m]/d[m]."""
        u = np.asarray(squared_norms, dtype=float)
        if u.shape != (self.n,) or not np.isfinite(u).all() or np.any(u < 0):
            raise ValueError("Squared primal norms must be finite nonnegative leaf values.")
        if not np.any(u):
            weights = np.ones(self.n)
            return weights / self.norm(weights)
        p1, p2, p3 = self.norms

        def aggregate(a, p):
            if np.isinf(p):
                return float(a.sum())
            return float(np.sum(a ** (p / (p + 1))) ** ((p + 1) / p))

        mu = [np.array([aggregate(u[g], p3) for g in groups]) for groups in self.groups]
        h = np.array([aggregate(m, p2) for m in mu])
        t = aggregate(h, p1)

        def distribute(a, total, p):
            if np.isinf(p):
                return np.ones_like(a)
            return (a / total) ** (1 / (p + 1)) if total > 0 else np.zeros_like(a)

        source_weights = distribute(h, t, p1)
        result = np.zeros(self.n)
        for j, groups in enumerate(self.groups):
            bias_weights = distribute(mu[j], h[j], p2) * source_weights[j]
            for s, indices in enumerate(groups):
                result[indices] = distribute(u[indices], mu[j][s], p3) * bias_weights[s]
        return result

    def constraints(self, weights):
        """Separate source constraints avoid differentiating a max at a tie."""
        _, sources, total = self._levels(np.asarray(weights))
        return 1 - sources if np.isinf(self.norms[0]) else np.array([1 - total])

    def constraint_jacobian(self, weights):
        weights = np.asarray(weights)
        p1, p2, p3 = self.norms
        biases, sources, _ = self._levels(weights)
        separate = np.isinf(p1)
        result = np.zeros((len(self.sources) if separate else 1, self.n))
        source_derivative = np.ones(len(sources)) if separate else _derivative(sources, p1)
        for j, groups in enumerate(self.groups):
            bias_derivative = _derivative(biases[j], p2)
            for s, indices in enumerate(groups):
                result[j if separate else 0, indices] = (
                    -source_derivative[j] * bias_derivative[s] * _derivative(weights[indices], p3)
                )
        return result
