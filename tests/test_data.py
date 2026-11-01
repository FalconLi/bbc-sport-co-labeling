import numpy as np
import pytest

from colabeling.data import ROOT, load_data, make_split


@pytest.mark.skipif(not (ROOT / "data/features/bbcsport.mtx").exists(), reason="Optional original dataset is not downloaded")
def test_original_protocol_and_strict_partitioning():
    dataset = load_data()
    split = make_split(dataset, seed=0)
    assert len(split.train) == 370
    assert len(split.test) == 367
    assert split.known_mask.sum() == 10
    assert not np.intersect1d(split.train, split.test).size
    np.testing.assert_array_equal(np.sort(np.r_[split.train, split.test]), np.arange(737))
    assert all((split.known_classes == category).sum() == 2 for category in range(5))
    assert np.all(split.known_classes[~split.known_mask] == -1)
    assert len(split.word_indices[0]) == 2306
    assert len(split.word_indices[1]) == 2307
    assert not np.intersect1d(*split.word_indices).size
    for view in split.train_views + split.test_views:
        sums = view.sum(axis=1)
        np.testing.assert_allclose(sums[sums > 0], 1, atol=1e-12)
    again = make_split(dataset, seed=0)
    np.testing.assert_array_equal(split.train, again.train)
    np.testing.assert_array_equal(split.word_indices[0], again.word_indices[0])
