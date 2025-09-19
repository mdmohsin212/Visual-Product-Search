from pymilvus import Collection
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys


class DatabaseSearch:
    def __init__(self, collection_name):
        try:
            self.collection = Collection(collection_name)
            self.collection.load()
            logging.info(f"Collection {collection_name} loaded for searching")
            
        except Exception as e:
            logging.error("Failed to load collection for search")
            raise ExceptionHandle(e, sys)
    
    def search(self, query_vec, k=5):
        try:
            logging.info("Performing search in Database")
            result = self.collection.search(
                data=query_vec,
                anns_field="embedding",
                param={"metric_type" : "COSINE", "params" : {"ef" : 32}},
                limit=k,
                output_fields=["metadata", "image_link"]
            )
            return result

        except Exception as e:
            logging.error("Search Failed in Database")
            raise ExceptionHandle(e, sys)