import json
from pathlib import Path
import pandas as pd

class EvaluationReportWriter:
    def __init__(self, config, root=None):
        self.config = config
        self.root = Path(root or Path.cwd())
        self.evaluation_config = config.get("evaluation", {})

    def resolve_path(self, value):
        path = Path(value)
        if path.is_absolute():
            return path

        return self.root / path

    def get_metrics_path(self):
        return self.resolve_path(
            self.evaluation_config.get(
                "metrics_output_path",
                "artifacts/evaluation/image_to_image_metrics.json",
            )
        )

    def get_category_path(self):
        return self.resolve_path(
            self.evaluation_config.get(
                "category_breakdown_output_path",
                "artifacts/evaluation/image_to_image_category_breakdown.csv",
            )
        )

    def get_error_path(self):
        return self.resolve_path(
            self.evaluation_config.get(
                "error_cases_output_path",
                "artifacts/evaluation/error_cases.csv",
            )
        )

    def save_json(self, data, path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, rows, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    def save(self, metrics_output, category_rows, error_cases):
        metrics_path = self.get_metrics_path()
        category_path = self.get_category_path()
        error_path = self.get_error_path()

        self.save_json(metrics_output, metrics_path)
        self.save_csv(category_rows, category_path)
        self.save_csv(error_cases, error_path)

        return {
            "metrics_path": str(metrics_path),
            "category_breakdown_path": str(category_path),
            "error_cases_path": str(error_path),
        }