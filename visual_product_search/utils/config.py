import sys
from pathlib import Path
import yaml
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle


def load_config(path: str = "config/model.yaml"):
    try:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")

        logging.info(f"Config loaded from path {config_path}")
        return config

    except Exception as e:
        logging.error(f"Error loading config {path}")
        raise ExceptionHandle(e, sys)