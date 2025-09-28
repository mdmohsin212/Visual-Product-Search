import torch, gc
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import os
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
            # self.env = environ.Env()
            # environ.Env.read_env(Path(__file__).resolve().parent.parent / ".env")
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
                lr=self.config["model"]["learning_rate"],
            )
            logging.info("Model training completed")
            return trained_model

        except Exception as e:
            raise ExceptionHandle(e, sys)
    
    def create_embeddings(self, dataloader, model, device):
        try:
            logging.info("Creating embeddings")
            embeddings = []
            img_link = []
            metadata = []
            
            with torch.no_grad():
                for batch in dataloader:
                    try:
                        imgs = batch["pixel_values"].to(device)
                        caps = batch["caption"]
                        links = batch["img_link"]

                        if imgs is None or len(imgs) == 0:
                            print("Image not found, skipping")
                            continue

                        img_embeds = model.get_image_features(pixel_values=imgs)
                        img_embeds = F(img_embeds, dim=-1)

                        embeddings.append(img_embeds.cpu())
                        metadata.extend(caps)
                        img_link.extend(links)

                    except Exception as e:
                        print(f"Skipping batch due to error: {e}")
                        continue

            if embeddings:
                embeddings = torch.cat(embeddings, dim=0)
            else:
                embeddings = torch.empty(0)
                
            logging.info(f"Embeddings shape: {embeddings.shape}")
            logging.info(f"Metadata length: {len(metadata)}")
            logging.info(f"Image links length: {len(img_link)}")
            return embeddings, metadata, img_link
        
        except Exception as e:
            raise ExceptionHandle(e, sys)

    def start_indexing(self, embeddings, metadata, img_link):
        try:
            logging.info("Starting indexing")
            indexer = DatabaseIndexer(
            uri=os.getenv("DATABASE_URL"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            token=os.getenv("TOKEN"),
            collection_name=os.getenv("COLLECTION_NAME")
            )
            indexer.insert_embeddings(embeddings, metadata, img_link)
            indexer.create_index()
            logging.info("index completed")
        
        except Exception as e:
            raise ExceptionHandle(e, sys)
    
    def push_hub(self, model, processor):
        try:
            logging.info("Starting Pushing model & processor in Hugginface")
            model.push_to_hub(
                self.config["model"]["new_model"],
                token=os.getenv("HF_TOKEN"),
                check_pr=False
            )
            processor.push_to_hub(
            self.config["model"]["new_model"],
            token=os.getenv("HF_TOKEN"),
            check_pr=False
            )
            logging.info("Model pushed to Hugging Face Hub")
        
        except Exception as e:
            logging.error("Model push Failed")
            raise ExceptionHandle(e, sys)
        
    def run_pipeline(self):
        try:
            df, img_dir = self.data_ingestion()
            model, processor, device = self.model_loading()
            
            dataset = ProductDataset(df, img_dir, processor, str(self.cache_dir))
            
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
            self.push_hub(trained_model, processor)
            
            embeddings, metadata, img_link = self.create_embeddings(dataloader, trained_model, device)
            self.start_indexing(embeddings, metadata, img_link)
            
            del model, processor, trained_model
            gc.collect()
            torch.cuda.empty_cache()    
            
        except Exception as e:
            logging.critical("Pipeline execution failed")
            raise ExceptionHandle(e, sys)


if __name__ == "__main__":
    pipeline = VisualProductPipeline()
    pipeline.run_pipeline()