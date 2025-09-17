from visual_product_search.data.ingest import load_data
from visual_product_search.data.dataset import ProductDataset, custom_collate_func
from visual_product_search.embeddings.model import load_model
from visual_product_search.embeddings.train import train
from visual_product_search.utils.config import load_config
from torch.utils.data import DataLoader
from logger import logging
from exception import ExceptionHandle
import sys


def main():
    try:
        config = load_config("config/model.yaml")
        df, img_dir = load_data()
        model, processor, device = load_model()
        
        dataset = ProductDataset(df, processor, img_dir)
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=custom_collate_func, shuffle=True)
        
        train(model, dataloader, device, epochs=5, lr=2e-5)
    
    except Exception as e:
        logging.critical("Pipeline Failed")
        raise ExceptionHandle(e, sys)
    
if __name__ == "__main__":
    main()