import numpy as np
import pytest

from colabeling.constraints import MILConstraints, RODConstraints, SSLConstraints


def test_ssl_keeps_known_labels_and_explores_eleven_balance_constraints():
    constraint = SSLConstraints(np.array([1, -1, -1, -1, 0, 0, 0, 0]))
    assert len(constraint.settings) == 11
    assert constraint.settings[0] == pytest.approx(.15)
    y = constraint.project(np.arange(8), .5)
    np.testing.assert_array_equal(y, [1, -1, -1, -1, -1, -1, 1, 1])


def test_mil_projects_each_positive_bag_and_never_relabels_negative_bags():
    c = MILConstraints(np.repeat(["negative", "positive-a", "positive-b"], 3), np.repeat([-1, 1, 1], 3))
    y = c.project(np.array([10., 9., 8., -3., -1., -2., 1., -1., 2.]), -.5)
    np.testing.assert_array_equal(y, [-1, -1, -1, -1, 1, -1, 1, -1, 1])
    np.testing.assert_array_equal(c.initial_labels, [-1, -1, -1, 1, 1, 1, 1, 1, 1])


def test_mil_portion_constraint_rounds_up_and_breaks_ties_by_instance_order():
    c = MILConstraints(np.array([0, 0, 0, 1, 1]), np.array([1, 1, 1, -1, -1]), positive_fraction=.5)
    np.testing.assert_array_equal(c.project(np.full(5, -2.), 0), [1, 1, -1, -1, -1])
    with pytest.raises(ValueError, match="same bag label"):
        MILConstraints(np.array([0, 0]), np.array([1, -1]))
    with pytest.raises(ValueError, match="fewer instances"):
        MILConstraints(np.array([0, 1]), np.array([1, -1]), minimum_positive=2)


def test_rod_uses_normal_positive_sign_and_fixed_reference_examples():
    c = RODConstraints(np.array([True, True, False, False, False, False]), .25)
    y = c.project(np.array([-100., -100., -3., 2., 1., 0.]), .25)
    np.testing.assert_array_equal(y, [1, 1, -1, 1, 1, 1])
    assert len(c.settings) == 11
    with pytest.raises(ValueError, match="outlier fraction"):
        RODConstraints(np.array([True, False]), 0)
