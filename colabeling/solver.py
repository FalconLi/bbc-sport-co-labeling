"""The paper's rho-SVM QP and two-layer MKL update, without pseudo-label voting.

    Q_m = (X X.T + 11.T) * (y_m y_m.T)
    alpha = argmin 0.5 alpha.T (sum_m d_m Q_m + I/C) alpha
            subject to alpha >= 0, sum(alpha) = 1.

For two views the view layer has one node. The grouped constraint becomes
    sqrt(sum_bias (sum_iteration d[bias, iteration])**2) <= 1.
Flat CoL(1-layer) uses sum(d) <= 1. Updates below specialize Eq. (10)-(17).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def project_labels(scores: np.ndarray, known: np.ndarray, positive_fraction: float) -> np.ndarray:
    """Eq. (3): keep supplied labels; give top-ranked unknowns +1.

    known is +1/-1 for supplied answers and 0 for unknown answers. An exact
    integer count uses round-half-up; tied scores use original document order.
    """
    scores, known = np.asarray(scores), np.asarray(known)
    if scores.ndim != 1 or scores.shape != known.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Scores and supplied labels must be matching finite vectors.")
    if not np.all(np.isin(known, [-1, 0, 1])) or not 0 <= positive_fraction <= 1:
        raise ValueError("Invalid labels or class proportion.")
    unknown = np.flatnonzero(known == 0)
    count = int(np.floor(positive_fraction * len(unknown) + 0.5))
    ranking = unknown[np.argsort(-scores[unknown], kind="stable")]
    answer = np.where(known == 0, -1, known).astype(np.int8)
    answer[ranking[:count]] = 1
    return answer


def mixed_norm(weights: np.ndarray, groups: np.ndarray, grouped: bool) -> float:
    totals = np.bincount(groups, weights=weights)
    return float(np.linalg.norm(totals) if grouped else weights.sum())


def update_weights(norm_squared: np.ndarray, groups: np.ndarray, grouped: bool) -> np.ndarray:
    """Closed-form minimizer of Eq. (11), specializing Eq. (13)-(17)."""
    if np.any(norm_squared < 0) or not np.all(np.isfinite(norm_squared)):
        raise ValueError("Primal squared norms must be finite and nonnegative.")
    roots = np.sqrt(norm_squared)
    if not grouped:
        return roots / roots.sum() if roots.sum() > 0 else np.full(len(roots), 1 / len(roots))
    group_roots = np.bincount(groups, weights=roots)
    a = group_roots**2
    group_weights = np.cbrt(a)
    if not np.any(group_weights):
        group_weights = np.ones_like(group_weights)
    group_weights /= np.linalg.norm(group_weights)
    within_group = np.divide(roots, group_roots[groups], out=np.zeros_like(roots), where=group_roots[groups] > 0)
    return within_group * group_weights[groups]


def _solve_qp_osqp(hessian: np.ndarray, warm: np.ndarray | None = None) -> np.ndarray:
    n = len(hessian)
    constraints = sparse.vstack([np.ones((1, n)), sparse.eye(n)], format="csc")
    problem = osqp.OSQP()
    problem.setup(
        P=sparse.csc_matrix(np.triu(hessian)), q=np.zeros(n), A=constraints,
        l=np.r_[1.0, np.zeros(n)], u=np.r_[1.0, np.full(n, np.inf)],
        verbose=False, eps_abs=1e-9, eps_rel=1e-8,
        max_iter=20_000, polishing=True, adaptive_rho=True,
    )
    if warm is not None:
        problem.warm_start(x=warm)
    result = problem.solve(raise_error=True)
    if result.info.status_val not in (1, 2) or result.x is None:
        raise RuntimeError(f"SVM optimization failed: {result.info.status}")
    alpha = np.maximum(result.x, 0)
    alpha /= alpha.sum()
    return alpha


def _solve_qp(hessian: np.ndarray, warm: np.ndarray | None = None) -> np.ndarray:
    """Solve the same strictly convex QP by a warm-started active set.

    Each free-set step solves H_PP z = 1. Bound hits preserve feasibility;
    reduced-gradient checks decide when to add an inactive variable. OSQP
    remains an independent fallback if the active-set iteration cap is hit.
    """
    n = len(hessian)
    tolerance = 1e-10 * max(1.0, float(np.abs(hessian).max()))
    if warm is None:
        z = cho_solve(cho_factor(hessian, check_finite=False), np.ones(n), check_finite=False)
        candidate = z / z.sum()
        if candidate.min() >= -1e-12:
            candidate = np.maximum(candidate, 0)
            return candidate / candidate.sum()
        free = candidate > 1e-12
        alpha = free.astype(float) / free.sum()
    else:
        alpha = np.maximum(warm, 0).copy()
        alpha /= alpha.sum()
        free = alpha > 1e-12
        alpha[~free] = 0
        alpha /= alpha.sum()
    for _ in range(3 * n + 20):
        indices = np.flatnonzero(free)
        z = cho_solve(cho_factor(hessian[np.ix_(indices, indices)], check_finite=False), np.ones(len(indices)), check_finite=False)
        target = np.zeros(n)
        target[indices] = z / z.sum()
        negative = free & (target < -1e-12)
        if np.any(negative):
            step = float(np.min(alpha[negative] / (alpha[negative] - target[negative])))
            alpha += step * (target - alpha)
            free[alpha <= 1e-12] = False
            alpha[~free] = 0
            alpha /= alpha.sum()
            continue
        alpha = np.maximum(target, 0)
        alpha /= alpha.sum()
        gradient = hessian @ alpha
        multiplier = float(alpha @ gradient)
        violations = np.where(~free, gradient - multiplier, np.inf)
        if violations.min(initial=np.inf) >= -tolerance:
            return alpha
        free[int(violations.argmin())] = True
    return _solve_qp_osqp(hessian, warm)


@dataclass
class KernelModel:
    alpha: np.ndarray
    weights: np.ndarray
    candidates: np.ndarray
    coefficients: np.ndarray
    rho: float
    objective: float
    gap: float
    inner_iterations: int
    converged: bool

    def decision_function(self, cross_kernel: np.ndarray) -> np.ndarray:
        # Rho normalization is explicit in the 2012 paper, Section IV-B.
        return cross_kernel @ self.coefficients / self.rho


def fit_mkl(
    kernel: np.ndarray,
    candidates: np.ndarray,
    groups: np.ndarray,
    *,
    c: float = 10.0,
    grouped: bool = True,
    max_inner: int = 100,
    tolerance: float = 1e-5,
) -> KernelModel:
    if c <= 0 or max_inner < 1 or tolerance <= 0:
        raise ValueError("C, max_inner, and tolerance must be positive.")
    candidates = np.asarray(candidates, dtype=np.float64)
    groups = np.asarray(groups, dtype=int)
    m, n = candidates.shape
    if kernel.shape != (n, n) or groups.shape != (m,) or not np.all(np.isin(candidates, [-1, 1])):
        raise ValueError("Invalid kernel or candidate dimensions/values.")
    if m < 1 or np.any(groups < 0):
        raise ValueError("At least one nonnegative group index is required.")
    weights = np.ones(m)
    weights /= mixed_norm(weights, groups, grouped)
    alpha = None
    for inner in range(1, max_inner + 1):
        # Schur product identity avoids materializing M dense Q_m matrices.
        combined = kernel * (candidates.T @ (weights[:, None] * candidates))
        hessian = combined + np.eye(n) / c
        alpha = _solve_qp(hessian, alpha)
        signed = candidates * alpha[None, :]
        energies = np.maximum(np.einsum("mi,mi->m", signed @ kernel, signed), 0)
        rho = float(alpha @ hessian @ alpha)
        objective = -0.5 * rho
        if grouped:
            best = np.zeros(groups.max() + 1)
            np.maximum.at(best, groups, energies)
            support = np.linalg.norm(best)
        else:
            support = energies.max()
        gap = max(0.0, 0.5 * float(support - weights @ energies))
        if gap <= tolerance * max(rho, 1e-12):
            break
        if inner < max_inner:
            weights = update_weights(weights**2 * energies, groups, grouped)
    # Multiplicative Eq. (15)-(17) updates approach sparse weights slowly.
    # If needed, polish the SAME convex MKL objective in the low-dimensional
    # coefficient space, using exact QP values and envelope gradients. This
    # changes the numerical solver, not the paper's objective/constraints.
    polished_iterations = 0
    if gap > tolerance * max(rho, 1e-12):
        scale = max(rho, 1e-12)
        cached_alpha = alpha

        def value_and_gradient(d):
            nonlocal cached_alpha
            matrix = kernel * (candidates.T @ (d[:, None] * candidates)) + np.eye(n) / c
            cached_alpha = _solve_qp(matrix, cached_alpha)
            signed = candidates * cached_alpha[None, :]
            energy = np.maximum(np.einsum("mi,mi->m", signed @ kernel, signed), 0)
            return -0.5 * float(cached_alpha @ matrix @ cached_alpha) / scale, -0.5 * energy / scale

        def constraint_gradient(d):
            if not grouped:
                return -np.ones(m)
            totals = np.bincount(groups, weights=d)
            return -totals[groups] / max(np.linalg.norm(totals), 1e-30)

        polished = minimize(
            value_and_gradient, weights, jac=True, method="SLSQP", bounds=[(0, None)] * m,
            constraints={"type": "ineq", "fun": lambda d: 1 - mixed_norm(d, groups, grouped), "jac": constraint_gradient},
            options={"ftol": min(1e-13, tolerance**2 * 0.01), "maxiter": max(100, 2 * max_inner)},
        )
        proposed = np.maximum(polished.x, 0)
        proposed /= max(mixed_norm(proposed, groups, grouped), 1e-30)
        value, _ = value_and_gradient(proposed)
        if value * scale <= objective + 1e-12:
            weights, alpha = proposed, cached_alpha
            combined = kernel * (candidates.T @ (weights[:, None] * candidates))
            hessian = combined + np.eye(n) / c
            rho = float(alpha @ hessian @ alpha)
            objective = -0.5 * rho
            signed = candidates * alpha[None, :]
            energies = np.maximum(np.einsum("mi,mi->m", signed @ kernel, signed), 0)
            if grouped:
                best = np.zeros(groups.max() + 1)
                np.maximum.at(best, groups, energies)
                support = np.linalg.norm(best)
            else:
                support = energies.max()
            gap = max(0.0, 0.5 * float(support - weights @ energies))
            polished_iterations = polished.nit
    coefficients = alpha * (weights @ candidates)
    return KernelModel(
        alpha, weights, candidates, coefficients, max(rho, np.finfo(float).tiny),
        objective, gap, inner + polished_iterations, gap <= tolerance * max(rho, 1e-12),
    )
