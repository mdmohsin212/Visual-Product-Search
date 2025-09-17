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

data = load_config('config/model.yaml')

print(data)