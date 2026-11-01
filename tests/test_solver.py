import numpy as np
import pytest
from scipy.optimize import minimize

from colabeling.solver import _solve_qp, _solve_qp_osqp, fit_mkl, mixed_norm, project_labels, update_weights


def test_projection_preserves_human_answers_and_exact_balance():
    scores = np.array([100., -100., 0., 0., 3., 2., 1.])
    known = np.array([-1, 1, 0, 0, 0, 0, 0])
    answer = project_labels(scores, known, 0.4)
    assert np.array_equal(answer[:2], known[:2])
    assert np.array_equal(np.flatnonzero(answer[2:] == 1), [2, 3])
    ties = project_labels(np.zeros(4), np.zeros(4), 0.5)
    assert np.array_equal(ties, [1, 1, -1, -1])


@pytest.mark.parametrize("grouped", [False, True])
def test_closed_form_weights_match_independent_constrained_optimizer(grouped):
    norms = np.array([0.8, 0.3, 2., 0.5, 1.2, 0.6])
    groups = np.array([0, 0, 1, 1, 2, 2])
    exact = update_weights(norms, groups, grouped)
    result = minimize(
        lambda d: float(np.sum(norms / d)), np.ones(6) / 6,
        method="SLSQP", bounds=[(1e-10, None)] * 6,
        constraints={"type": "ineq", "fun": lambda d: 1 - mixed_norm(d, groups, grouped)},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    assert result.success
    np.testing.assert_allclose(exact, result.x, rtol=2e-5, atol=2e-6)
    assert mixed_norm(exact, groups, grouped) == pytest.approx(1)


def test_simplex_qp_matches_independent_optimizer():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(8, 5))
    hessian = features @ features.T + np.eye(8) / 3
    alpha = _solve_qp(hessian)
    reference = minimize(
        lambda a: 0.5 * a @ hessian @ a, np.full(8, 1 / 8),
        jac=lambda a: hessian @ a, bounds=[(0, None)] * 8,
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1},
        method="SLSQP", options={"ftol": 1e-12},
    )
    np.testing.assert_allclose(alpha, reference.x, atol=1e-6)


def test_single_labeling_recovers_standard_rho_svm_and_prediction_scale():
    x = np.array([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]])
    y = np.array([-1., -1., 1., 1.])
    kernel = x @ x.T + 1
    model = fit_mkl(kernel, y[None, :], np.array([0]), c=2)
    expected_alpha = _solve_qp(kernel * np.outer(y, y) + np.eye(4) / 2)
    np.testing.assert_allclose(model.alpha, expected_alpha)
    assert model.converged
    scores = model.decision_function(kernel)
    np.testing.assert_array_equal(np.sign(scores), y)
    active = model.alpha > 1e-7
    np.testing.assert_allclose(y[active] * scores[active] + model.alpha[active] / (2 * model.rho), 1, atol=1e-6)


def test_active_set_matches_osqp_across_bound_hits_and_warm_starts():
    rng = np.random.default_rng(123)
    previous = None
    for _ in range(25):
        x = rng.normal(size=(14, 7))
        h = x @ x.T + np.eye(14) * 0.03
        answer = _solve_qp(h, previous)
        reference = _solve_qp_osqp(h)
        np.testing.assert_allclose(answer, reference, atol=2e-6)
        assert np.min(answer) >= 0
        assert answer.sum() == pytest.approx(1)
        previous = answer


@pytest.mark.parametrize("grouped", [False, True])
def test_mkl_reaches_global_saddle_gap_tolerance(grouped):
    rng = np.random.default_rng(20)
    x = rng.normal(size=(16, 5))
    kernel = x @ x.T + 1
    candidates = rng.choice([-1, 1], size=(9, 16))
    groups = np.repeat(np.arange(3), 3)
    model = fit_mkl(kernel, candidates, groups, c=3, grouped=grouped, max_inner=50)
    assert model.converged
    assert mixed_norm(model.weights, groups, grouped) == pytest.approx(1, abs=1e-6)
