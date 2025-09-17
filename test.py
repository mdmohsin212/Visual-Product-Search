from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

try:
    n = 3 / 0
    print(n)
    
except Exception as e:
    logging.error(e)
    raise ExceptionHandle(e, sys)