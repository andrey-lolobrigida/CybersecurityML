from src.main import Pipeline, run


def test_build_pipeline_type(pipeline):
    assert isinstance(pipeline, Pipeline)
    assert pipeline.cfg is not None
    assert pipeline.model is not None


def test_run_returns_expected_keys(raw_row, pipeline):
    result = run(raw_row, pipeline)
    assert set(result.keys()) == {"prediction", "label", "probability"}


def test_run_prediction_is_binary(raw_row, pipeline):
    result = run(raw_row, pipeline)
    assert result["prediction"] in (0, 1)


def test_run_probability_in_range(raw_row, pipeline):
    result = run(raw_row, pipeline)
    assert 0.0 <= result["probability"] <= 1.0


def test_run_label_matches_prediction(raw_row, pipeline):
    result = run(raw_row, pipeline)
    expected = {0: "Normal session", 1: "Possible attack session"}
    assert result["label"] == expected[result["prediction"]]


def test_run_null_encryption_used(raw_row, pipeline):
    raw_row["encryption_used"] = None
    result = run(raw_row, pipeline)
    assert result["prediction"] in (0, 1)


def test_run_without_session_id(raw_row, pipeline):
    raw_row.pop("session_id")
    result = run(raw_row, pipeline)
    assert result["prediction"] in (0, 1)