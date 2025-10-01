from visual_product_search.pipeline.training_pipeline import VisualProductPipeline
from visual_product_search.pipeline.prediction_pipeline import ProductPredictionPipeline
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle

from flask import Flask, render_template, request
import sys

app = Flask(__name__)
predictionPipeline = ProductPredictionPipeline()

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/train", methods=["GET"])
def model_train():
    try:
        pipeline = VisualProductPipeline()
        pipeline.run_pipeline()
        return "Training completed successfully"
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise ExceptionHandle(e, sys)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        k = int(request.form.get("k", 5))
               
        if request.form.get("text_field"):
            query = request.form["text_field"]
            outputs = predictionPipeline.search_with_text(query, k)
            results = [item.entity['image_link'] for item in outputs[0]]
            return render_template("home.html", results=results)

        elif "img_field" in request.files:
            img_file = request.files["img_field"]   
            outputs = predictionPipeline.search_with_image(img_file, k)
            results = [item.entity['image_link'] for item in outputs[0]]
            return render_template("home.html", results=results)

        else:
            return render_template("home.html", result="No input provided")

    except Exception as e:
        logging.critical(f"Prediction Failed: {e}")
        raise ExceptionHandle(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)