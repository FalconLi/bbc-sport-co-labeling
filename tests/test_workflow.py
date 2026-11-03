import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from colabeling import CoLabelingClassifier
from colabeling.data import ROOT
from colabeling.io import load_dataset, save_dataset
from colabeling.metrics import classification_metrics, evaluate_model
from make_example_data import make_examples


def command(*args):
    env = os.environ.copy()
    for name in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
        env[name] = "1"
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=90)


@pytest.mark.parametrize("task", ["ssl", "mil", "rod"])
def test_real_train_save_predict_evaluate_commands(tmp_path, task):
    data_dir = make_examples(tmp_path / "data") / task
    model_path = tmp_path / f"{task}.npz"
    args = ["train.py", "--task", task, "--input", data_dir / "train.npz", "--model", model_path, "--max-rounds", "4"]
    if task == "rod":
        args += ["--outlier-fraction", ".25"]
    trained = command(*args)
    assert trained.returncode == 0, trained.stderr
    model = CoLabelingClassifier.load(model_path)
    views, data = load_dataset(data_dir / "test.npz")
    # Prediction must also work when no test answers are present in the input.
    inputs = tmp_path / "features-only.npz"
    save_dataset(inputs, views, **({"bag_ids": data["bag_ids"]} if task == "mil" else {}))
    predicted = command("predict.py", "--model", model_path, "--input", inputs, "--output", tmp_path / "predictions.npz")
    assert predicted.returncode == 0, predicted.stderr
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as result:
        expected = model.predict_bags(views, data["bag_ids"])[1] if task == "mil" else model.predict(views)
        np.testing.assert_array_equal(result["labels"], expected)
        assert np.isfinite(result["scores"]).all()
    evaluated = command("evaluate.py", "--model", model_path, "--input", data_dir / "test.npz", "--output", tmp_path / "metrics.json")
    assert evaluated.returncode == 0, evaluated.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["accuracy"] >= 80
    assert metrics["map" if task == "ssl" else "ap"] >= 80
    assert metrics.get("unit") == ("bag" if task == "mil" else "instance")
    missing = command("evaluate.py", "--model", model_path, "--input", inputs, "--output", tmp_path / "missing.json")
    assert missing.returncode != 0 and "true test labels" in missing.stderr


def test_sparse_dataset_interchange_and_missing_view_validation(tmp_path):
    views = [sparse.csr_matrix(np.eye(5)), np.eye(5)]
    save_dataset(tmp_path / "data.npz", views, y=np.array(["a", "b", "a", "b", "a"]), known_mask=np.ones(5, dtype=bool))
    loaded, data = load_dataset(tmp_path / "data.npz")
    assert sparse.issparse(loaded[0])
    np.testing.assert_array_equal(loaded[0].toarray(), np.eye(5))
    assert data["y"].dtype.kind == "U"
    np.savez(tmp_path / "bad.npz", view_0=np.eye(2), view_2=np.eye(2))
    with pytest.raises(ValueError, match="contiguous"):
        load_dataset(tmp_path / "bad.npz")


def test_map_handles_arbitrary_labels_and_absent_test_classes():
    labels = np.array(["a", "b", "a", "b"])
    scores = np.array([[3, 0, -1], [0, 3, -1], [3, 0, -1], [0, 3, -1]])
    report = classification_metrics(labels, np.stack([scores, scores]), np.array(["a", "b", "c"]))
    assert report["accuracy"] == 100
    assert report["ap_by_class"] == [100, 100, 0]
    assert report["map"] == pytest.approx(200 / 3)
    with pytest.raises(ValueError, match="absent"):
        classification_metrics(np.array(["d"] * 4), scores[None], np.array(["a", "b", "c"]))


def test_bag_evaluation_rejects_inconsistent_bag_labels(tmp_path):
    make_examples(tmp_path)
    views, data = load_dataset(tmp_path / "mil/train.npz")
    model = CoLabelingClassifier(task="mil", max_rounds=2).fit(views, data["y"], bag_ids=data["bag_ids"])
    labels = data["y"].copy(); labels[1] = "positive"
    with pytest.raises(ValueError, match="consistent"):
        evaluate_model(model, views, labels, bag_ids=data["bag_ids"])
