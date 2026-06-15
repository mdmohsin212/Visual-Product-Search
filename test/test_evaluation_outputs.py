import json
from pathlib import Path


def test_evaluation_metrics_file_exists():
    path = Path("artifacts/evaluation/image_to_image_metrics.json")
    assert path.exists(), "image_to_image_metrics.json is missing"


def test_evaluation_metrics_has_required_keys():
    path = Path("artifacts/evaluation/image_to_image_metrics.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "evaluation_type" in data
    assert "model" in data
    assert "dataset" in data
    assert "num_queries" in data
    assert "num_indexed_images" in data
    assert "metrics" in data


def test_evaluation_metrics_has_relevance_levels():
    path = Path("artifacts/evaluation/image_to_image_metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    assert "strict" in metrics
    assert "medium" in metrics
    assert "soft" in metrics


def test_evaluation_csv_files_exist():
    category_path = Path("artifacts/evaluation/image_to_image_category_breakdown.csv")
    error_path = Path("artifacts/evaluation/error_cases.csv")

    assert category_path.exists(), "image_to_image_category_breakdown.csv is missing"
    assert error_path.exists(), "error_cases.csv is missing"