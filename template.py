import os
from pathlib import Path

project_name = "visual_product_search"

list_of_files = [
    f"{project_name}/embeddings/__init__.py",
    f"{project_name}/embeddings/model.py",
    f"{project_name}/embeddings/embed.py",
    f"{project_name}/embeddings/train.py",
    f"{project_name}/indexing/__init__.py",
    f"{project_name}/indexing/indexer.py",
    f"{project_name}/indexing/faiss_utils.py",
    f"{project_name}/data/__init__.py",
    f"{project_name}/data/ingest.py",
    f"{project_name}/data/dataset.py",
    f"{project_name}/monitoring/__init__.py",
    f"{project_name}/monitoring/prometheus_exporter.py",
    f"{project_name}/monitoring/grafana_dashboard.json",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/config.py",
    f"{project_name}/utils/storage.py",
    f"{project_name}/__main__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_api.py",
    f"{project_name}/tests/test_indexing.py",
    f"{project_name}/tests/test_embeddings.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
    "test.py",
    "dvc.yaml",
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
