"""The offline replay must agree with the experiment it claims to show."""

import json

import numpy as np
import pytest

from colabeling.data import ROOT
from colabeling.experiment import evaluate


def read_recording():
    text = (ROOT / "demo/demo-data.js").read_text()
    return json.loads(text.removeprefix("window.BBCSPORT_DEMO = ").removesuffix(";\n"))


def test_recording_has_separate_test_articles_and_only_ten_supplied_answers():
    data = read_recording()
    articles = data["articles"]
    assert len({article["id"] for article in articles}) == 737
    training = [a for a in articles if a["role"] == "train"]
    testing = [a for a in articles if a["role"] == "test"]
    assert len(training) == 370 and len(testing) == 367
    assert sum(a["known"] for a in training) == 10
    assert not any(a["known"] for a in testing)
    assert data["recording_seed"] == data["experiment"]["seeds"][0]


def test_saved_scores_reproduce_revealed_predictions_and_evaluation():
    data = read_recording()
    testing = sorted((a for a in data["articles"] if a["role"] == "test"), key=lambda a: a["row"])
    answers = np.array([a["answer"] for a in testing])
    for method, recording in data["methods"].items():
        for state in recording["states"]:
            scores = np.array(state["test_views"])
            assert scores.shape == (2, 367, 5)
            assert np.isfinite(scores).all()
            np.testing.assert_array_equal(scores.mean(axis=0).argmax(axis=1), state["test_predictions"])
            actual = evaluate(answers, scores)
            assert actual["map"] == pytest.approx(state["metrics"]["map"], abs=1e-4)
        final = recording["states"][-1]["metrics"]["map"]
        reported = data["experiment"]["runs"][0]["methods"][method]["map"]
        assert final == pytest.approx(reported, abs=1e-8)
