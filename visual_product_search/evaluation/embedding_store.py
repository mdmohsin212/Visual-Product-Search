from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from visual_product_search.embeddings.model import load_model


class EmbeddingStore:
    def __init__(self, config, root=None):
        self.config = config
        self.root = Path(root or Path.cwd())

        self.model_config = config.get("model", {})
        self.data_config = config.get("data", {})
        self.embedding_config = config.get("embeddings", {})
        self.index_config = config.get("index", {})

    def resolve_path(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root / path

    def normalize_embeddings(self, embeddings):
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def get_embedding_path(self):
        return self.resolve_path(
            self.embedding_config.get(
                "output_path",
                self.index_config.get(
                    "embedding_path",
                    "artifacts/embeddings/image_embeddings.npy",
                ),
            )
        )

    def get_metadata_aligned_path(self):
        return self.resolve_path(
            self.data_config.get(
                "metadata_aligned_path",
                "artifacts/embeddings/metadata_aligned.csv",
            )
        )

    def load_existing(self, metadata):
        embedding_path = self.get_embedding_path()
        if not embedding_path.exists() or embedding_path.stat().st_size == 0:
            return None, metadata

        embeddings = np.load(embedding_path)
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embedding rows ({len(embeddings)}) and metadata rows ({len(metadata)}) do not match"
            )

        return self.normalize_embeddings(embeddings), metadata

    def get_model_name(self):
        return (
            self.model_config.get("repo_id")
            or self.model_config.get("new_model")
            or self.model_config.get("name")
        )

    def get_device(self):
        device = self.model_config.get("device", "auto")
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    def build_embeddings(self, metadata):
        model_name = self.get_model_name()
        device = self.get_device()
        batch_size = int(
            self.embedding_config.get(
                "batch_size",
                self.model_config.get("batch_size", 32),
            )
        )

        model, processor, device = load_model(model_name, device=device)
        embeddings = []
        valid_rows = []
        image_paths = metadata["image_path"].fillna("").astype(str).tolist()

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = []
            rows = []

            for offset, image_path in enumerate(batch_paths):
                path = Path(image_path)
                if not path.exists():
                    continue

                try:
                    images.append(Image.open(path).convert("RGB"))
                    rows.append(metadata.iloc[start + offset])
                except Exception:
                    continue

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.inference_mode():
                batch_embeddings = model.get_image_features(**inputs)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)

            embeddings.append(batch_embeddings.cpu().numpy())
            valid_rows.extend(rows)

        if not embeddings:
            raise RuntimeError("No image embeddings could be generated. Check image paths.")

        embeddings = np.vstack(embeddings).astype(np.float32)
        aligned_metadata = pd.DataFrame(valid_rows).reset_index(drop=True)

        embedding_path = self.get_embedding_path()
        metadata_aligned_path = self.get_metadata_aligned_path()

        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_aligned_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(embedding_path, embeddings)
        aligned_metadata.to_csv(metadata_aligned_path, index=False)

        return self.normalize_embeddings(embeddings), aligned_metadata

    def load_or_build(self, metadata):
        embeddings, metadata = self.load_existing(metadata)
        if embeddings is not None:
            return embeddings, metadata

        return self.build_embeddings(metadata)