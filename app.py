import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from visual_product_search.pipeline.training_pipeline import VisualProductPipeline
from visual_product_search.pipeline.prediction_pipeline import ProductPredictionPipeline
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
from visual_product_search.utils.config import load_json_file, to_percent

app = Flask(__name__)
prediction_pipeline = None

def get_prediction_pipeline():
    global prediction_pipeline
    
    if prediction_pipeline is None:
        prediction_pipeline = ProductPredictionPipeline()
    
    return prediction_pipeline


def prepare_evaluation_view(metrics_data):
    if not metrics_data or metrics_data.get("status") == "not_run_yet":
        return None
    
    metrics = metrics_data.get("metrics", {})
    rows = [
        {
            "name": "Article Type Match",
            "definition": "Same articleType",
            "source": "strict",
        },
        {
            "name": "Subcategory Match",
            "definition": "Same subCategory",
            "source": "medium",
        },
        {
            "name": "Gender + Master Category Match",
            "definition": "Same gender and masterCategory",
            "source": "soft",
        },
    ]

    prepared_rows = []
    for row in rows:
        source_metrics = metrics.get(row["source"], {})

        prepared_rows.append(
            {
                "name": row["name"],
                "definition": row["definition"],
                "precision_5": to_percent(source_metrics.get("precision@5", 0)),
                "ndcg_10": to_percent(source_metrics.get("ndcg@10", 0)),
                "map_10": to_percent(source_metrics.get("map@10", 0)),
                "mrr_10": to_percent(source_metrics.get("mrr@10", 0)),
            }
        )

    return {
        "model": metrics_data.get("model", "N/A"),
        "dataset": metrics_data.get("dataset", "N/A"),
        "num_queries": metrics_data.get("num_queries", 0),
        "num_indexed_images": metrics_data.get("num_indexed_images", 0),
        "embedding_dimension": metrics_data.get("embedding_dimension", 0),
        "exclude_self_match": metrics_data.get("exclude_self_match", False),
        "rows": prepared_rows,
        "main_cards": {
            "article_precision_5": prepared_rows[0]["precision_5"],
            "subcategory_precision_5": prepared_rows[1]["precision_5"],
            "gender_master_precision_5": prepared_rows[2]["precision_5"],
        },
    }


def prepare_benchmark_view(benchmark_data):
    if not benchmark_data or benchmark_data.get("status") == "not_run_yet":
        return None

    latency = benchmark_data.get("latency_ms", {})
    artifact_sizes = benchmark_data.get("artifact_sizes", {})

    return {
        "benchmark_type": benchmark_data.get("benchmark_type", "N/A"),
        "top_k": benchmark_data.get("top_k", 0),
        "num_runs": benchmark_data.get("num_runs", 0),
        "mean": latency.get("mean", 0),
        "p50": latency.get("p50", 0),
        "p95": latency.get("p95", 0),
        "p99": latency.get("p99", 0),
        "embedding_file_mb": artifact_sizes.get("embedding_file_mb", 0),
        "metadata_file_mb": artifact_sizes.get("metadata_file_mb", 0),
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "visual-product-search",
        }
    )

@app.route("/evaluation", methods=["GET"])
def evaluation_dashboard():
    metrics_data = load_json_file("artifacts/evaluation/image_to_image_metrics.json")
    benchmark_data = load_json_file("artifacts/benchmark/benchmark_results.json")

    evaluation_view = prepare_evaluation_view(metrics_data)
    benchmark_view = prepare_benchmark_view(benchmark_data)

    return render_template(
        "evaluation.html",
        evaluation=evaluation_view,
        benchmark=benchmark_view,
    )

@app.route("/train", methods=["GET"])
def train_page():
    return render_template("train.html")

@app.route("/train_model", methods=["GET"])
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
        pipeline = get_prediction_pipeline()
        
        if request.form.get("text_field"):
            query = request.form["text_field"]
            outputs = pipeline.search_with_text(query, k)
            results = [item.entity['image_link'] for item in outputs[0]]
            return render_template("home.html", results=results)

        elif "img_field" in request.files:
            img_file = request.files["img_field"]   
            outputs = pipeline.search_with_image(img_file, k)
            results = [item.entity['image_link'] for item in outputs[0]]
            return render_template("home.html", results=results)

        else:
            return render_template("home.html", result="No input provided")

    except Exception as e:
        logging.critical(f"Prediction Failed: {e}")
        raise ExceptionHandle(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)