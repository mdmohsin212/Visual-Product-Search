import torch
from torch.nn.functional import normalize
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

def get_text_embedding(model : CLIPModel, processor : CLIPProcessor, text, device="cpu"):
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            txt_emb = model.get_text_features(**inputs)
            txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)
            
        logging.debug(f"Image embedding generated for {text}")
        return txt_emb.cpu().numpy().tolist()[0]
    
    except Exception as e:
        logging.error(f"Error embedding text : {text}")
        raise ExceptionHandle(e, sys)


def get_image_embedding(model : CLIPModel, processor : CLIPProcessor, image_path, device="cpu"):
    try:
        image = Image.open(image_path).resize((224,224))
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
            img_emb = torch.nn.functional.normalize(img_emb, dim=-1)

        return img_emb.cpu().numpy().tolist()[0]
    
    except Exception as e:
        logging.error(f"Error embedding image : {image_path}")
        raise ExceptionHandle(e, sys)