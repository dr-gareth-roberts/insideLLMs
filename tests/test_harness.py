import yaml

from insideLLMs.runner import run_harness_from_config


def test_run_harness_from_config(tmp_path):
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text('{"question": "What is 2 + 2?"}\n{"question": "What is 3 + 3?"}\n')

    config = {
        "models": [{"type": "dummy", "args": {}}],
        "probes": [{"type": "logic", "args": {}}],
        "dataset": {"format": "jsonl", "path": str(dataset_path)},
        "max_examples": 1,
    }

    config_path = tmp_path / "harness.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_harness_from_config(config_path)

    assert len(result["records"]) == 1
    assert len(result["experiments"]) == 1
    assert result["summary"]["total_experiments"] == 1

    record = result["records"][0]
    # Harness should emit standard ResultRecord-shaped records.
    assert "run_id" in record
    assert "model" in record and isinstance(record["model"], dict)
    assert "probe" in record and isinstance(record["probe"], dict)
    assert "dataset" in record and isinstance(record["dataset"], dict)
    assert record.get("custom", {}).get("harness", {}).get("experiment_id")


def test_run_harness_from_config_validate_output(tmp_path):
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text('{"question": "What is 2 + 2?"}\n')

    config = {
        "models": [{"type": "dummy", "args": {}}],
        "probes": [{"type": "logic", "args": {}}],
        "dataset": {"format": "jsonl", "path": str(dataset_path)},
        "max_examples": 1,
    }

    config_path = tmp_path / "harness.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # Should validate records against standard ResultRecord schema.
    result = run_harness_from_config(config_path, validate_output=True)
    assert len(result["records"]) == 1
