from torch.utils.data import DataLoader
import numpy as np
import environ
from pathlib import Path
import sys

from visual_product_search.data.ingest import load_data
from visual_product_search.data.dataset import ProductDataset
from visual_product_search.embeddings.model import load_model
from visual_product_search.embeddings.train import train
from visual_product_search.utils.config import load_config
from visual_product_search.embeddings.embed import get_image_embedding
from visual_product_search.indexing.indexer import DatabaseIndexer
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle


class VisualProductPipeline:
    def __init__(self, config_path="config/model.yaml"):
        try:
            self.env = environ.Env()
            environ.Env.read_env(Path(__file__).resolve().parent.parent / ".env")
            self.config = load_config(config_path)
            
            self.cache_dir = Path("cache_dir")
            self.cache_dir.mkdir(exist_ok=True)
            logging.info(f"Preprocessed images will be cached in {self.cache_dir}")
        
        except Exception as e:
            logging.critical("Failed to load config or environment")
            raise ExceptionHandle(e, sys)
    
    def data_ingestion(self):
        try:
            logging.info("Start data ingestion")
            df, img_dir = load_data()
            return df, img_dir
        
        except Exception as e:
            raise ExceptionHandle(e, sys)
    
    def model_loading(self):
        try:
            logging.info("Loading model")
            model, processor, device = load_model(self.config["model"]["name"])
            return model, processor, device

        except Exception as e:
            raise ExceptionHandle(e, sys)
    
    def start_training(self, model, dataloader, device):
        try:
            logging.info("Start model training")
            trained_model = train(
                model,
                dataloader,
                device,
                epochs=self.config["model"]["epochs"],
                lr=self.config["model"]["learning_rate"]
            )
            logging.info("Model training completed")
            return trained_model

        except Exception as e:
            raise ExceptionHandle(e, sys)
    
    def create_embeddings(self, df, img_dir, model, processor, device):
        try:
            logging.info("Creating embeddings")
            embeddings = []
            metadata = []
            
            for row in df.itertuples(index=False):
                img_path = f"{img_dir}/{row.filename}"
                embd =  get_image_embedding(img_path, model, processor, device)
                embeddings.append(embd)
                filtered_row = row._asdict()
                filtered_row.pop("filename", None)
                filtered_row.pop("link", None)
                metadata.append(" ".join(str(v) for v in filtered_row.values()))
            
            embeddings = np.vstack(embeddings)
            logging.info(f"create embeddings for {len(df)} samples")
            return embeddings, metadata
        
        except Exception as e:
            raise ExceptionHandle(e, sys)

    def start_indexing(self, embeddings, metadata):
        try:
            logging.info("Starting indexing")
            indexer = DatabaseIndexer(
            uri=self.env("DATABASE_URL"),
            user=self.env("USER"),
            password=self.env("PASSWORD"),
            token=self.env("TOKEN"),
            collection_name=self.env("COLLECTION_NAME")
            )
            indexer.insert_embeddings(embeddings, metadata)
            indexer.create_index()
            logging.info("index completed")
        
        except Exception as e:
            raise ExceptionHandle(e, sys)
        
    def run_pipeline(self):
        try:
            df, img_dir = self.data_ingestion()
            model, processor, device = self.model_loading()
            
            dataset = ProductDataset(df, processor, img_dir, str(self.cache_dir))
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config["model"]["batch_size"],
                shuffle=True,
                num_workers=12,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            trained_model = self.start_training(model, dataloader, device)
            
            embeddings, metadata = self.create_embeddings(df, img_dir, trained_model, processor, device)
            self.start_indexing(embeddings, metadata)
            
        except Exception as e:
            logging.critical("Pipeline execution failed")
            raise ExceptionHandle(e, sys)


if __name__ == "__main__":
    pipeline = VisualProductPipeline()
    pipeline.run_pipeline()