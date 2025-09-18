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
    def __init__(self, df: pd.DataFrame, processor: CLIPProcessor, img_dir: str, preprocessed_dir: str = None):
        self.df = df
        self.processor = processor
        self.img_dir = img_dir
        self.preprocessed_dir = preprocessed_dir

        if preprocessed_dir:
            os.makedirs(preprocessed_dir, exist_ok=True)
            logging.info(f"Preprocessed images will be cached in {preprocessed_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_name = row['filename']
            img_path = os.path.join(self.img_dir, img_name)

            tensor_path = os.path.join(self.preprocessed_dir, img_name + ".pt")
                
            if os.path.exists(tensor_path):
                pixel_values = torch.load(tensor_path)
                    
            else:
                image = Image.open(img_path).resize((224, 224))
                pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                torch.save(pixel_values, tensor_path)

            filtered_row = row.drop(labels=["filename", "link"])
            text = " ".join(str(v) for v in filtered_row.values)
            text_inputs = self.processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            input_ids = text_inputs["input_ids"].squeeze(0)
            attention_mask = text_inputs["attention_mask"].squeeze(0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values
            }

        except Exception as e:
            logging.error(f"Error processing item {idx}")
            raise ExceptionHandle(e, sys)