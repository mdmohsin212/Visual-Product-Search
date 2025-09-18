from visual_product_search.pipeline.training_pipeline import VisualProductPipeline
from visual_product_search.logger import logging

def run_pipeline_thread():
    try:
        pipeline = VisualProductPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        
run_pipeline_thread()