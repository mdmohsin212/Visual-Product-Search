from visual_product_search.data.ingest import load_data
from visual_product_search.data.dataset import ProductDataset, custom_collate_func
from visual_product_search.embeddings.model import load_model
from visual_product_search.embeddings.train import train
from visual_product_search.utils.config import load_config
from visual_product_search.embeddings.embed import get_image_embedding, get_text_embedding
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle

from torch.utils.data import DataLoader
import numpy as np
import sys


def main():
    try:
        config = load_config("config/model.yaml")
        
        df, img_dir = load_data()
        model, processor, device = load_model(config["model"]["name"])
        
        dataset = ProductDataset(df, processor, img_dir)
        dataloader = DataLoader(dataset, batch_size=config["model"]["batch_size"], collate_fn=custom_collate_func, shuffle=True)
        
        train(model, dataloader, device, epochs=config["model"]["epochs"], lr=config["model"]["learning_rate"])
        
    except Exception as e:
        logging.critical("Pipeline Failed")
        raise ExceptionHandle(e, sys)
    
if __name__ == "__main__":
    main()