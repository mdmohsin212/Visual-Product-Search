from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys
from visual_product_search.utils.config import load_config

# try:
#     n = 3 / 0
#     print(n)
    
# except Exception as e:
#     logging.error(e)
#     raise ExceptionHandle(e, sys)

# data = load_config('config/model.yaml')

# print(data)

from pymilvus import connections
# import environ

# env = environ.Env()
# environ.Env.read_env()

# print(env("USER"))


# from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

# collection_name = "fashion_clip_embeddings"



# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="e", dtype=DataType.FLOAT_VECTOR, dim=768),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
# ]

# schema = CollectionSchema(fields, description="Fashion CLIP embeddings")
# collection = Collection(name=collection_name, schema=schema)

# import numpy as np

# n_samples = 2000
# embeddings = [np.random.rand(768).astype(np.float32) for _ in range(n_samples)]
# texts = [f"Product description {i}" for i in range(n_samples)]


# batch_size = 500
# n = len(embeddings)

# for start in range(0, n, batch_size):
#     end = min(start + batch_size, n)
#     batch_emb = embeddings[start:end]
#     batch_text = texts[start:end]

#     entities = [
#         batch_emb,
#         batch_text  
#     ]
#     insert_result = collection.insert(entities)
#     collection.flush()
#     print(f"Inserted {len(batch_emb)} rows ({start} → {end})")


# print(f"Inserted total {n} embeddings")
# print("Total entities:", collection.num_entities)

# import requests 
# API_URL = "https://api-inference.huggingface.co/models/mohsin416/clip-vit-large-patch14-fashion-retrieval" 
# HF_TOKEN = ""
# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} 
# def get_text_embedding(text): 
#     payload = {"inputs": text} 
#     response = requests.post(API_URL, headers=HEADERS, json=payload) 
#     return response.json()

# get_text_embedding("hello")

