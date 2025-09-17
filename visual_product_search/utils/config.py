import sys
import yaml
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle

def load_config(path : str):
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Config loaded from path {path}")
        return config

    except Exception as e:
        logging.erorr(f"Error loading config {path}")
        raise ExceptionHandle(e, sys)