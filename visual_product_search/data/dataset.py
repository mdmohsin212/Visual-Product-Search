import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor
from visual_product_search.exception import ExceptionHandle
from visual_product_search.logger import logging


class ProductDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_folder: str, processor: CLIPProcessor, preprocessed_dir: str = None):
        try:
            df["image_path"] = df["filename"].astype(str).apply(
                lambda x: os.path.join(image_folder, x)
            )
            df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

            df["caption"] = (
                df["gender"].fillna("") + " "
                + df["masterCategory"].fillna("") + " "
                + df["subCategory"].fillna("") + " "
                + df["baseColour"].fillna("") + " "
                + df["articleType"].fillna("") + " "
                + df["productDisplayName"].fillna("")
            ).str.strip()

            self.df = df.reset_index(drop=True)
            self.processor = processor
            self.preprocessed_dir = preprocessed_dir

            if preprocessed_dir:
                os.makedirs(preprocessed_dir, exist_ok=True)
                logging.info(f"Preprocessed images will be cached in {preprocessed_dir}")

        except Exception as e:
            raise ExceptionHandle(e, sys)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = row["image_path"]

            tensor_path = (
                os.path.join(self.preprocessed_dir, row["filename"] + ".pt")
                if self.preprocessed_dir
                else None
            )

            if tensor_path and os.path.exists(tensor_path):
                pixel_values = torch.load(tensor_path)
            else:
                try:
                    image = Image.open(img_path).resize((224, 224))
                    pixel_values = self.processor(images=image, return_tensors="pt")[
                        "pixel_values"
                    ].squeeze(0)
                    if tensor_path:
                        torch.save(pixel_values, tensor_path)
                except FileNotFoundError:
                    logging.warning(f"Image not found: {img_path}, using dummy tensor.")
                    pixel_values = torch.zeros(3, 224, 224)

            caption = row["caption"]
            img_link = row.get("link", None)

            text_inputs = self.processor.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )

            input_ids = text_inputs["input_ids"].squeeze(0)
            attention_mask = text_inputs["attention_mask"].squeeze(0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "caption": caption,
                "img_link": img_link,
            }

        except Exception as e:
            logging.error(f"Error processing dataset item {idx}")
            raise ExceptionHandle(e, sys)