from pathlib import Path
import pandas as pd
from visual_product_search.data.ingest import load_data

class EvaluationDataLoader:
    def __init__(self, config, root=None):
        self.config = config
        self.root = Path(root or Path.cwd())
        self.data_config = config.get("data", {})

    def resolve_path(self, value):
        path = Path(value)
        if path.is_absolute():
            return path

        return self.root / path

    def infer_image_path(self, row, image_dir):
        image_dir = Path(image_dir)
        candidates = []

        for column in ["image_path", "filename", "image", "id"]:
            if column in row.index and not pd.isna(row[column]):
                value = str(row[column]).strip()
                candidates.append(value)

                if column == "id" and not value.endswith(".jpg"):
                    candidates.append(f"{value}.jpg")

        for value in candidates:
            path = Path(value)
            if path.exists():
                return str(path)

            joined = image_dir / value
            if joined.exists():
                return str(joined)

            joined_images = image_dir / "images" / value
            if joined_images.exists():
                return str(joined_images)

        return ""

    def load_metadata_from_local(self):
        metadata_path = self.resolve_path(
            self.data_config.get("metadata_path", "artifacts/styles.csv")
        )

        image_dir = self.resolve_path(
            self.data_config.get("image_dir", "artifacts/images")
        )

        if not metadata_path.exists() or metadata_path.stat().st_size == 0:
            return None

        df = pd.read_csv(metadata_path, on_bad_lines="skip")
        if "image_path" not in df.columns:
            df["image_path"] = df.apply(
                lambda row: self.infer_image_path(row, image_dir),
                axis=1,
            )

        df = df[df["image_path"].astype(str).str.len() > 0].reset_index(drop=True)
        return df

    def load_metadata_from_aligned(self):
        metadata_aligned_path = self.resolve_path(
            self.data_config.get(
                "metadata_aligned_path",
                "artifacts/embeddings/metadata_aligned.csv",
            )
        )

        if not metadata_aligned_path.exists() or metadata_aligned_path.stat().st_size == 0:
            return None

        return pd.read_csv(metadata_aligned_path)

    def load_metadata_from_dataset(self):
        df, downloaded_image_dir = load_data()
        if "image_path" not in df.columns:
            df["image_path"] = df.apply(
                lambda row: self.infer_image_path(row, downloaded_image_dir),
                axis=1,
            )

        df = df[df["image_path"].astype(str).str.len() > 0].reset_index(drop=True)
        return df

    def save_aligned_metadata(self, metadata):
        metadata_aligned_path = self.resolve_path(
            self.data_config.get(
                "metadata_aligned_path",
                "artifacts/embeddings/metadata_aligned.csv",
            )
        )

        metadata_aligned_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_aligned_path, index=False)

    def load(self):
        aligned_metadata = self.load_metadata_from_aligned()
        if aligned_metadata is not None:
            return aligned_metadata

        local_metadata = self.load_metadata_from_local()
        if local_metadata is not None:
            self.save_aligned_metadata(local_metadata)
            return local_metadata

        dataset_metadata = self.load_metadata_from_dataset()
        self.save_aligned_metadata(dataset_metadata)

        return dataset_metadata