import sys
import environ
from pathlib import Path
import os
from pymilvus import connections

from visual_product_search.embeddings.embed import get_image_embedding, get_text_embedding
from visual_product_search.indexing.search import DatabaseSearch
from visual_product_search.utils.config import load_config
from visual_product_search.embeddings.model import load_model_for_prediction
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle

class ProductPredictionPipeline:
    _model = None
    _processor = None
    _device = None
    _Database = None
    def __init__(self, config_path="config/model.yaml"):
        try:
            # self.env = environ.Env()
            # environ.Env.read_env(Path(__file__).resolve().parent.parent / ".env")
            self.config = load_config(config_path)
            
            connections.connect(alias="default", 
                uri=os.getenv("DATABASE_URL"),
                user=os.getenv("USER"),
                password=os.getenv("PASSWORD"),
                token=os.getenv("TOKEN"),
            )
            
            if ProductPredictionPipeline._Database is None:
                ProductPredictionPipeline._Database = DatabaseSearch(
                    collection_name=os.getenv("COLLECTION_NAME")
                )

        except Exception as e:
            logging.critical("Failed to load config, environment or model")
            raise ExceptionHandle(e, sys)
    
    def _load_model(self):
        if ProductPredictionPipeline._model is None:
            ProductPredictionPipeline._model, ProductPredictionPipeline._processor, ProductPredictionPipeline._device = load_model_for_prediction(self.config["model"]["name"], self.config["model"]["lora_model"])
            logging.info("Model loaded successfully.")
    
    def search_with_image(self, image_path, k=5):
        try:
            self._load_model()
            print("Start searching using image")
            embd = get_image_embedding(
                ProductPredictionPipeline._model,
                ProductPredictionPipeline._processor,
                image_path,
                ProductPredictionPipeline._device
            )
            results = ProductPredictionPipeline._Database.search([embd], k)
            print("Successfully found items from database")
            return results

        except Exception as e:
            print("Searching failed")
            raise e


    def search_with_text(self, text: str, k=5):
        try:
            self._load_model()
            print("Start searching using text")
            embd = get_text_embedding(
                ProductPredictionPipeline._model,
                ProductPredictionPipeline._processor,
                text,
                ProductPredictionPipeline._device
            )
            results = ProductPredictionPipeline._Database.search([embd], k)
            print("Successfully found items from database")
            return results

        except Exception as e:
            print("Searching failed")
            raise e