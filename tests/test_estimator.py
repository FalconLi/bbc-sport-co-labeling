import numpy as np
import pytest
from scipy import sparse

from colabeling import CoLabelingClassifier


def ssl_data(views=3):
    rng = np.random.default_rng(17)
    classes = np.array(["athletics-like", "football-like", "tennis-like"])
    encoded = np.repeat(np.arange(3), 10)
    features = [np.eye(3)[encoded] * 3 + rng.normal(0, .2, (30, 3)) for _ in range(views)]
    mask = np.tile([True, True] + [False] * 8, 3)
    return features, classes[encoded], mask


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_arbitrary_views_string_classes_history_and_serialization(tmp_path, layers):
    views, y, mask = ssl_data(views=4)
    views[2] = np.column_stack([views[2], np.ones(30)])
    model = CoLabelingClassifier(layers=layers, max_rounds=3, record_history=True).fit(views, y, known_mask=mask)
    assert model.decision_function_views(views).shape == (4, 30, 3)
    assert (model.predict(views) == y).mean() > .9
    assert all(s["source_view"] != v for run in model.kernel_weights_ for v, sheets in enumerate(run) for s in sheets)
    for run in model.kernel_weights_:
        for sheets in run:
            assert len({s["source_view"] for s in sheets}) == 3
    model.save(tmp_path / "model.npz")
    restored = CoLabelingClassifier.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(restored.predict(views), model.predict(views))
    np.testing.assert_allclose(restored.decision_function_views(views), model.decision_function_views(views), atol=1e-12)
    with np.load(tmp_path / "model.npz", allow_pickle=False) as archive:
        assert all(archive[k].dtype.kind != "O" for k in archive.files)


def test_unknown_answers_do_not_enter_training_and_arbitrary_numeric_labels_work():
    views, y, mask = ssl_data()
    first = CoLabelingClassifier(max_rounds=2).fit(views, y, known_mask=mask)
    changed = y.copy(); changed[~mask] = "unseen-hidden-label"
    second = CoLabelingClassifier(max_rounds=2).fit(views, changed, known_mask=mask)
    np.testing.assert_array_equal(first.classes_, second.classes_)
    np.testing.assert_allclose(first.decision_function(views), second.decision_function(views), atol=1e-12)
    numeric = np.array([-7, 19, 83])[np.repeat(np.arange(3), 10)]
    third = CoLabelingClassifier(max_rounds=2).fit(views, numeric, known_mask=mask)
    np.testing.assert_array_equal(third.classes_, [-7, 19, 83])


def test_rbf_sparse_per_view_normalization_and_roundtrip(tmp_path):
    views, y, mask = ssl_data()
    views = [sparse.csr_matrix(np.maximum(x, 0)) for x in views]
    model = CoLabelingClassifier(kernel=["linear", "rbf", "rbf"], normalization="l1", max_rounds=2).fit(views, y, known_mask=mask)
    assert model.kernels_[1].gamma > 0
    scores = model.decision_function(views)
    assert np.isfinite(scores).all()
    model.save(tmp_path / "sparse.npz")
    restored = CoLabelingClassifier.load(tmp_path / "sparse.npz")
    np.testing.assert_allclose(restored.decision_function(views), scores, atol=1e-12)


def test_mil_uses_bag_initialization_and_predicts_bags(tmp_path):
    rng = np.random.default_rng(7)
    bags = np.repeat(np.arange(8), 3)
    bag_y = np.repeat(np.array(["negative", "positive"] * 4), 3)
    x = np.full((24, 2), -2.) + rng.normal(0, .1, (24, 2))
    x[np.flatnonzero((bag_y == "positive") & (np.arange(24) % 3 == 0))] += 4
    views = [x, x + rng.normal(0, .05, x.shape), x[:, ::-1]]
    model = CoLabelingClassifier(task="mil", max_rounds=4, record_history=True).fit(views, bag_y, bag_ids=bags)
    assert model.history_[0]["initialization"] == "bag_labels"
    expected = np.where(bag_y == "positive", 1, -1)
    np.testing.assert_array_equal(model.train_scores_[0][0, :, 1], expected)
    ids, predicted = model.predict_bags(views, bags)
    np.testing.assert_array_equal(ids, np.arange(8))
    assert len(predicted) == 8
    with pytest.raises(ValueError, match="initialization"):
        model.decision_function_views(views, round_index=0)
    model.save(tmp_path / "mil.npz")
    restored = CoLabelingClassifier.load(tmp_path / "mil.npz")
    np.testing.assert_array_equal(restored.predict_bags(views, bags)[1], predicted)


@pytest.mark.parametrize("normal_label,outlier_label", [("normal", "outlier"), (7, -8)])
def test_rod_normal_only_initialization_and_outlier_score_direction(tmp_path, normal_label, outlier_label):
    rng = np.random.default_rng(9)
    normal = rng.normal(2, .15, (20, 3))
    outlier = rng.normal(-2, .15, (5, 3))
    x = np.r_[normal, outlier]
    reference = np.arange(25) < 5
    views = [x, x + rng.normal(0, .05, x.shape), x[:, ::-1]]
    model = CoLabelingClassifier(task="rod", outlier_fraction=.25, max_rounds=3,
                                normal_label=normal_label, outlier_label=outlier_label).fit(views, known_normal_mask=reference)
    assert model.history_[0]["initialization"] == "reference_normal_one_class_svm"
    scores = model.decision_function(views)
    assert scores[-5:, 1].mean() > scores[:20, 1].mean()
    assert (model.predict(views) == np.array([normal_label] * 20 + [outlier_label] * 5)).mean() > .9
    model.save(tmp_path / "rod.npz")
    np.testing.assert_array_equal(CoLabelingClassifier.load(tmp_path / "rod.npz").predict(views), model.predict(views))


def test_alignment_and_task_inputs_are_validated():
    views, y, mask = ssl_data()
    with pytest.raises(ValueError, match="same samples"):
        CoLabelingClassifier().fit([views[0], views[1][:-1]], y, known_mask=mask)
    with pytest.raises(ValueError, match="at least two views"):
        CoLabelingClassifier().fit([views[0]], y, known_mask=mask)
    with pytest.raises(ValueError, match="at least two observed classes|At least two observed classes"):
        CoLabelingClassifier().fit(views, np.zeros(30), known_mask=mask)
    model = CoLabelingClassifier(max_rounds=1).fit(views, y, known_mask=mask)
    with pytest.raises(ValueError, match="feature dimension"):
        model.predict([views[0][:, :2], *views[1:]])
    with pytest.raises(ValueError, match="training outlier labels"):
        CoLabelingClassifier(task="rod").fit(views, y, known_normal_mask=mask)


def test_explicit_ssl_priors_control_the_candidate_constraints():
    views, labels, mask = ssl_data()
    model = CoLabelingClassifier(max_rounds=1, class_priors=[.5, .3, .2]).fit(views, labels, known_mask=mask)
    assert model.kernel_weights_[0][0][0]["fraction"] == pytest.approx(.4)
    assert model.kernel_weights_[1][0][0]["fraction"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="summing to one"):
        CoLabelingClassifier(class_priors=[.2, .2, .2]).fit(views, labels, known_mask=mask)
