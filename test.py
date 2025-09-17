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
import environ

env = environ.Env()
environ.Env.read_env()

print(env("USER"))


# from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

# collection_name = "fashion_clip_embeddings"

# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
#     FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024)
# ]

# schema = CollectionSchema(fields, description="Fashion CLIP embeddings")
# collection = Collection(name=collection_name, schema=schema)
