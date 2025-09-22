from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

class DatabaseIndexer:
    def __init__(self, uri, user, password, token, collection_name):
        try:
            connections.connect(alias="default", uri=uri, user=user, password=password, token=token)
            
            logging.info("Connect to Zilliz Cloud")
            self.connection_name = collection_name
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="image_link", dtype=DataType.VARCHAR, max_length=2048)
            ]
            
            schema = CollectionSchema(fields, description="Fashion Embedding")
            
            if utility.has_collection(self.connection_name):
                self.collection = Collection(self.connection_name)
                print(f"Loaded existing collection: {self.connection_name}")
            else:
                self.collection = Collection(name=self.connection_name, schema=schema)
                print(f"Created new collection: {self.connection_name}")
                
        except Exception as e:
            logging.error("Failed to initialize Database")
            raise ExceptionHandle(e, sys)
    
    def insert_embeddings(self, embeddings, metadata, img_link, batch_size=500):
        try:
            entities = [embeddings, metadata, img_link]
            self.collection.insert(entities)
            self.collection.flush()
            logging.info(f"Inserted total {len(embeddings)} embeddings ")
        
        except Exception as e:
            logging.error("Failed to insert embeddings")
            raise ExceptionHandle(e, sys)
    
    def create_index(self):
            try:
                index_params = {
                    "index_type" : "HNSW",
                    "metric_type" : "COSINE",
                    "params" : {"M" : 8, "efConstruction" : 64}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                logging.info("Index Created successfully")
            
            except Exception as e:
                logging.error("Failed to create index")
                raise ExceptionHandle(e, sys)