"""Check the animation against the learner's recorded values, not its artwork."""

import numpy as np
import pytest

from colabeling.data import ROOT
from colabeling.solver import project_labels
from presentation_data import CHAPTERS, fingerprints, read_demo, read_story


def test_story_predictions_and_changes_match_every_saved_round():
    demo, story = read_demo(), read_story()
    unknown = sorted((a for a in demo["articles"] if a["role"] == "train" and not a["known"]), key=lambda a: a["row"])
    assert story["article_ids"] == [a["id"] + 1 for a in unknown]
    assert len(story["article_ids"]) == 360
    for i, actual in enumerate(story["rounds"]):
        original = demo["methods"]["colabel_grouped"]["states"][i]
        expected = [original["train_predictions"][a["row"]] for a in unknown]
        assert actual["predictions"] == expected
        old = story["rounds"][max(0, i - 1)]["predictions"]
        assert actual["changed_count"] == sum(a != b for a, b in zip(expected, old))
    assert [r["changed_count"] for r in story["rounds"]] == [0, 98, 35, 9, 5, 2, 1, 0, 0]


def test_every_displayed_sheet_is_the_real_projection_of_its_source_reader():
    demo, story = read_demo(), read_story()
    training = sorted((a for a in demo["articles"] if a["role"] == "train"), key=lambda a: a["row"])
    unknown_rows = [a["row"] for a in training if not a["known"]]
    known = np.array([0 if not a["known"] else (1 if a["answer"] == 2 else -1) for a in training])
    assert len(story["sheets"]) == 22
    for sheet in story["sheets"]:
        state = demo["methods"]["colabel_grouped"]["states"][sheet["origin_round"]]
        scores = np.array(state["train_views"])[sheet["source_view"], :, 2]
        expected = project_labels(scores, known, sheet["fraction"])[unknown_rows]
        np.testing.assert_array_equal(expected, sheet["labels"])
        assert sheet["football_count"] == int((expected == 1).sum())
        assert sheet["preview"] == [int(expected[a["index"]]) for a in story["preview"]]


def test_final_influence_matches_saved_optimizer_and_has_no_invented_values():
    demo, story = read_demo(), read_story()
    snapshots = story["optimizer"]["snapshots"]
    expected = demo["methods"]["colabel_grouped"]["states"][2]["details"]["categories"][2]["views"][0]["sheets"]
    np.testing.assert_allclose(snapshots[-1]["weights"], [s["weight"] for s in expected], rtol=1e-7, atol=1e-9)
    assert snapshots[-1]["label"] == "Final solution"
    assert len({tuple(s["weights"]) for s in snapshots}) > 2
    for snapshot in snapshots:
        weights = np.array(snapshot["weights"])
        assert np.isfinite(weights).all() and (weights >= 0).all()
        np.testing.assert_allclose(snapshot["shares"], 100 * weights / weights.sum())
    assert story["source_sha256"] == fingerprints()


def test_chart_statistics_are_aggregated_from_all_ten_runs():
    demo, story = read_demo(), read_story()
    assert story["evaluation_runs"] == 10 and story["test_articles"] == 367
    for result in story["results"]:
        values = [r["methods"][result["method"]]["map"] for r in demo["experiment"]["runs"]]
        assert result["map_mean"] == pytest.approx(np.mean(values))
        assert result["map_std"] == pytest.approx(np.std(values, ddof=1))


def test_shared_scene_timing_covers_exactly_ninety_seconds():
    story = read_story()
    assert story["duration"] == 90
    assert CHAPTERS[0][1] == 0 and CHAPTERS[-1][2] == 90
    assert all(first[2] == second[1] for first, second in zip(CHAPTERS, CHAPTERS[1:]))
    assert [(c["id"], c["start"], c["end"], c["title"]) for c in story["chapters"]] == CHAPTERS


def test_mp4_is_playable_ninety_seconds_and_contains_changing_weight_bars():
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    movie = ROOT / "demo/training-animation.mp4"
    assert movie.exists()
    frames = ffmpeg.read_frames(str(movie), pix_fmt="rgb24")
    try:
        metadata = next(frames)
        assert metadata["size"] == (1600, 900)
        assert metadata["duration"] == pytest.approx(90, abs=.05)
    finally:
        frames.close()
    crops = []
    for second in [36, 58]:
        frames = ffmpeg.read_frames(str(movie), pix_fmt="rgb24", input_params=["-ss", str(second)])
        try:
            metadata = next(frames)
            image = np.frombuffer(next(frames), dtype=np.uint8).reshape(900, 1600, 3)
            crops.append(image[349:669, 320:1360].astype(float))
        finally:
            frames.close()
    assert np.abs(crops[0] - crops[1]).mean() > 5, "Weight bars must actually change in the video."
