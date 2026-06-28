import json
import sys
from visual_product_search.evaluation.evaluator import ImageToImageEvaluator
from visual_product_search.exception import ExceptionHandle

def main():
    try:
        evaluator = ImageToImageEvaluator()
        result = evaluator.run()
        print(json.dumps(result, indent=2))

    except Exception as e:
        raise ExceptionHandle(e, sys)


if __name__ == "__main__":
    main()