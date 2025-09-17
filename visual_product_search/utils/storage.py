import os
import pickle
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

def save_pickle(obj, path : str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved at {path}")
    except Exception as e:
        raise ExceptionHandle(e, sys)
    
def load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logging.info(f"Object loaded from {path}")
        return obj
    except Exception as e:
        raise ExceptionHandle(e, sys)