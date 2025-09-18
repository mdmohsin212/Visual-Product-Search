import kagglehub
import pandas as pd
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

def load_data():
    try:
        logging.info("Downloading dataset from KaggleHub..")
        path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
        
        logging.info("Loading Metadata CSV..")
        df = pd.read_csv('https://raw.githubusercontent.com/mdmohsin212/Machine-Learning/refs/heads/main/dataset/fashion_product.csv')
        
        logging.info(f"Dataset load with {len(df)}")
        
        return df, f"{path}/fashion-dataset/images"

    except Exception as e:
        logging.error("Failed during dataset ingestion")
        raise ExceptionHandle(e, sys)
