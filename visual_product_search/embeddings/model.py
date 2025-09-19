import torch
from transformers import CLIPModel, CLIPProcessor
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

def load_model(model_name : str, device : str = None):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading CLIP model {model_name} on {device}")
        
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        logging.info("Model and processor loaded successfully")
        return model, processor, device
    
    except Exception as e:
        logging.error("Model loading failed")
        raise ExceptionHandle(e, sys)