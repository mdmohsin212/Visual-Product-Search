import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
from transformers import CLIPProcessor
from PIL import Image
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import pandas as pd
import sys, os

class ProductDataset(Dataset):
    def __init__(self, df : pd.DataFrame, processor : CLIPProcessor, img_dir : str):
        self.df = df
        self.processor = processor
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row['filename'])
            if not os.path.exists(img_path):
                logging.warning(f"Image not found : {img_path}")
                raise FileNotFoundError(f"Image missing: {img_path}")
            
            image = Image.open(img_path).resize((224, 224))
            filtered_row = row.drop(labels=["filename", "link"])
            text = " ".join(str(v) for v in filtered_row.values)
            
            inputs = self.processor(text=[text], images=image, padding=True, return_tensors="pt")

            return {
                "input_ids" : inputs['input_ids'].squeeze(0),
                "attention_mask" : inputs['attention_mask'].squeeze(0),
                "pixel_values" : inputs['pixel_values'].squeeze(0),
            }
            
        except Exception as e:
            logging.error(f"Error in item {idx}")
            raise ExceptionHandle(e, sys)


def custom_collate_func(batch, processor : CLIPProcessor):
    try:
        max_len = max([len(item['input_ids']) for item in batch])

        padded_input_ids = torch.stack([
            pad(item['input_ids'], (0, max_len - len(item['input_ids'])),
            value=processor.tokenizer.pad_token_id) for item in batch
        ])

        padded_attention_mask = torch.stack([
            pad(item['attention_mask'], (0, max_len - len(item['attention_mask'])),
            value=processor.tokenizer.pad_token_id) for item in batch
        ])

        pixel_values = torch.stack([item['pixel_values'] for item in batch])

        return{
            "input_ids" : padded_input_ids,
            "attention_mask" : padded_attention_mask,
            "pixel_values" : pixel_values
        }
    
    except Exception as e:
        logging.error("Collate Function Failed")
        ExceptionHandle(e, sys)