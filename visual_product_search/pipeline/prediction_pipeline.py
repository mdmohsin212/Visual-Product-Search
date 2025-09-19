import torch
import sys
import environ
from pathlib import Path

from visual_product_search.embeddings.embed import get_image_embedding, get_text_embedding
from visual_product_search.indexing.search import DatabaseSearch
from visual_product_search.utils.config import load_config
from visual_product_search.embeddings.model import load_model
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle

class ProductPredictionPipeline:
    def __init__(self, config_path="config/model.yaml"):
        try:
            self.env = environ.Env()
            environ.Env.read_env(Path(__file__).resolve().parent.parent / ".env")
            self.config = load_config(config_path)
            self.model, self.processor, self.device = load_model(self.config["mode"]["new_model"])
        
        except Exception as e:
            logging.critical("Failed to load config, environment or model")
            raise ExceptionHandle(e, sys)
