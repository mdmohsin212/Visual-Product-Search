import torch
from torch.nn.functional import normalize
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

def get_image_embedding(image_path : str, model : CLIPModel, processor : CLIPProcessor, device="cuda"):
    try:
        image = Image.open(image_path).resize((224, 224))
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            embd = model.get_image_features(**inputs)
            embd = normalize(embd, p=2, dim=-1)
        
        logging.debug(f"Image embedding generated for {image_path}")
        return embd.cpu().numpy()
    
    except Exception as e:
        logging.warning(f"Image not found or unreadable: {image_path}. Using dummy embedding.")
        try:
            dummy_embd = torch.randn(1, model.config.projection_dim, device=device)
            dummy_embd = normalize(dummy_embd, p=2, dim=-1)
            return dummy_embd.cpu().numpy()
        
        except Exception as inner_e:
            raise ExceptionHandle(inner_e, sys)
        
    
def get_text_embedding(text : str, model : CLIPModel, processor : CLIPProcessor, device="cuda"):
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            embd = model.get_text_features(**inputs)
            embd = normalize(embd, p=2, dim=-1)
        
        logging.debug(f"Text embedding generated for: {text[:10]}...")
        return embd.cpu().numpy()
    
    except Exception as e:
        logging.error(f"Error embedding text : {text}")
        raise ExceptionHandle(e, sys)
    