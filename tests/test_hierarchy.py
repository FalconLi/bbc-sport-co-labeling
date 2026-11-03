import numpy as np
import pytest
from scipy.optimize import minimize

from colabeling.hierarchy import KernelHierarchy
from colabeling.solver import fit_mkl


@pytest.mark.parametrize("norms", [(np.inf, 2, 1), (2, 2, 1), (3, 2, 1.5), (1, 1, 1)])
def test_recursive_three_layer_update_matches_independent_optimizer(norms):
    sources = np.array([0, 0, 0, 0, 4, 4, 4, 4, 4])
    biases = np.array([0, 0, 2, 2, 0, 0, 2, 2, 3])
    tree = KernelHierarchy(sources, biases, norms)
    u = np.array([.8, .3, 2., .5, 1.2, .6, .4, .7, .9])
    exact = tree.update(u)
    initial = np.ones(9); initial /= tree.norm(initial)
    reference = minimize(lambda d: np.sum(u / d) / u.sum(), initial,
                         jac=lambda d: -u / d**2 / u.sum(), method="SLSQP", bounds=[(1e-9, None)] * 9,
                         constraints={"type": "ineq", "fun": tree.constraints, "jac": tree.constraint_jacobian},
                         options={"ftol": 1e-11, "maxiter": 1000})
    assert reference.success, reference.message
    np.testing.assert_allclose(exact, reference.x, rtol=3e-5, atol=3e-6)
    assert tree.norm(exact) == pytest.approx(1)


@pytest.mark.parametrize("norms", [(np.inf, 2, 1), (2, 2, 1), (3, 2, 1.5)])
def test_hierarchy_constraint_gradient_and_dual_support(norms):
    source, bias = np.repeat([0, 1], 6), np.tile(np.repeat([0, 1, 2], 2), 2)
    tree = KernelHierarchy(source, bias, norms)
    d = np.linspace(.1, 1, 12)
    numeric = np.column_stack([(tree.constraints(d + np.eye(12)[i] * 1e-6)
                                - tree.constraints(d - np.eye(12)[i] * 1e-6)) / 2e-6 for i in range(12)])
    np.testing.assert_allclose(tree.constraint_jacobian(d), numeric, atol=2e-6)
    energies = np.linspace(.2, 2, 12)
    initial = np.ones(12); initial /= tree.norm(initial)
    result = minimize(lambda w: -energies @ w, initial, jac=lambda w: -energies,
                      method="SLSQP", bounds=[(0, None)] * 12,
                      constraints={"type": "ineq", "fun": tree.constraints, "jac": tree.constraint_jacobian},
                      options={"ftol": 1e-12, "maxiter": 1000})
    assert result.success
    assert tree.support(energies) == pytest.approx(-result.fun, abs=1e-6)


@pytest.mark.parametrize("norms", [(np.inf, 2, 1), (2, 2, 1)])
def test_three_layer_solver_satisfies_global_saddle_gap(norms):
    rng = np.random.default_rng(33)
    x = rng.normal(size=(18, 6))
    candidates = rng.choice([-1, 1], (12, 18))
    source, bias = np.repeat([0, 2], 6), np.tile(np.repeat([0, 1, 2], 2), 2)
    tree = KernelHierarchy(source, bias, norms)
    model = fit_mkl(x @ x.T + 1, candidates, bias, source_groups=source,
                    norms=norms, max_inner=80, c=3)
    assert model.converged
    assert tree.norm(model.weights) == pytest.approx(1, abs=1e-6)
    assert model.gap <= 1e-5 * max(-2 * model.objective, 1e-12)


def test_source_identities_and_zero_primal_norms_are_handled():
    tree = KernelHierarchy(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert tree.norm(np.ones(4)) == pytest.approx(np.sqrt(2))
    # Merging identically numbered biases across readers would give sqrt(8).
    assert tree.norm(tree.update(np.zeros(4))) == pytest.approx(1)
    original = tree.update(np.array([1., 2., 3., 4.]))
    order = np.array([2, 0, 3, 1])
    reordered = KernelHierarchy(np.array([0, 0, 1, 1])[order], np.array([0, 1, 0, 1])[order])
    np.testing.assert_allclose(reordered.update(np.array([1., 2., 3., 4.])[order]), original[order])
