import os
from pathlib import Path

project_name = "visual_product_search"

list_of_files = [
    f"{project_name}/data/__init__.py",
    f"{project_name}/data/ingest.py",
    f"{project_name}/data/dataset.py",
    
    f"{project_name}/embeddings/__init__.py",
    f"{project_name}/embeddings/model.py",
    f"{project_name}/embeddings/embed.py",
    f"{project_name}/embeddings/train.py",
    
    f"{project_name}/evaluation/__init__.py",
    f"{project_name}/evaluation/metrics.py",
    f"{project_name}/evaluation/evaluate_image_to_image.py",
    f"{project_name}/evaluation/error_analysis.py",
    
    f"{project_name}/exception/__init__.py",
    
    f"{project_name}/indexing/__init__.py",
    f"{project_name}/indexing/indexer.py", 
    f"{project_name}/indexing/search.py",
    
    f"{project_name}/logger/__init__.py",
    
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/config.py",
    f"{project_name}/utils/storage.py",
    
    "artifacts/evaluation/image_to_image_metrics.json",
    "artifacts/evaluation/image_to_image_category_breakdown.csv",
    "artifacts/evaluation/error_cases.csv",
    
    ".github/workflows/docker-to-hf.yml",
    
    "config/model.yaml",
    "config/schema.yaml",
    
    "templates/home.html",
    "templates/train.html",
    
    ".dockerignore",
    ".gitignore",
    "app.py",
    "Dockerfile",
    "LICENCE",
    "README.md",
    "requirements.txt",
    "setup.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
    else:
        print(f'{filename} is already present in {filedir} and has some content. Skipping creation.')