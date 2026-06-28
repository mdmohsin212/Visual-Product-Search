from collections import defaultdict
from pathlib import Path
import numpy as np
from visual_product_search.evaluation.data_loader import EvaluationDataLoader
from visual_product_search.evaluation.embedding_store import EmbeddingStore
from visual_product_search.evaluation.error_analysis import build_error_case
from visual_product_search.evaluation.metrics import average_metric_dicts, evaluate_ranking
from visual_product_search.evaluation.relevance import RelevanceComputer
from visual_product_search.evaluation.report import EvaluationReportWriter
from visual_product_search.evaluation.similarity import SimilaritySearcher
from visual_product_search.logger import logging
from visual_product_search.utils.config import load_config


class ImageToImageEvaluator:
    def __init__(self, config_path="config/model.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.root = Path.cwd()

        self.model_config = self.config.get("model", {})
        self.data_config = self.config.get("data", {})
        self.evaluation_config = self.config.get("evaluation", {})

        self.k_values = [int(k) for k in self.evaluation_config.get("k_values", [1, 5, 10])]
        self.max_k = max(self.k_values)

        self.query_batch_size = int(self.evaluation_config.get("query_batch_size", 64))
        self.max_queries = self.evaluation_config.get("max_queries")
        self.seed = int(self.evaluation_config.get("seed", 42))
        self.exclude_self_match = bool(self.evaluation_config.get("exclude_self_match", True))

        self.data_loader = EvaluationDataLoader(self.config, self.root)
        self.embedding_store = EmbeddingStore(self.config, self.root)
        self.report_writer = EvaluationReportWriter(self.config, self.root)

    def select_query_indices(self, total_items):
        indices = np.arange(total_items)
        if self.max_queries is None:
            return indices

        max_queries = int(self.max_queries)
        if max_queries <= 0 or max_queries >= total_items:
            return indices

        rng = np.random.default_rng(self.seed)
        selected = rng.choice(indices, size=max_queries, replace=False)

        return np.sort(selected)

    def get_category_field(self, metadata):
        if "articleType" in metadata.columns:
            return "articleType"

        if "subCategory" in metadata.columns:
            return "subCategory"

        return None

    def build_metrics_output(self, metadata, embeddings, query_indices, relevance_computer, metric_storage):
        return {
            "evaluation_type": "image_to_image_retrieval",
            "model": (
                self.model_config.get("repo_id")
                or self.model_config.get("new_model")
                or self.model_config.get("name")
            ),
            "dataset": self.data_config.get(
                "dataset_name",
                "paramaggarwal/fashion-product-images-dataset",
            ),
            "num_queries": int(len(query_indices)),
            "num_indexed_images": int(len(metadata)),
            "embedding_dimension": int(embeddings.shape[1]),
            "k_values": self.k_values,
            "exclude_self_match": self.exclude_self_match,
            "relevance_definitions": relevance_computer.levels,
            "metrics": {
                level: average_metric_dicts(values)
                for level, values in metric_storage.items()
            },
        }

    def build_category_rows(self, category_storage):
        category_rows = []
        for level_name, categories in category_storage.items():
            for category, values in categories.items():
                row = {
                    "relevance_level": level_name,
                    "category": category,
                    "num_queries": len(values),
                }

                row.update(average_metric_dicts(values))
                category_rows.append(row)

        return category_rows

    def run(self):
        metadata = self.data_loader.load()
        embeddings, metadata = self.embedding_store.load_or_build(metadata)

        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length")

        query_indices = self.select_query_indices(len(metadata))

        relevance_computer = RelevanceComputer(metadata, self.evaluation_config)
        searcher = SimilaritySearcher(
            top_k=self.max_k,
            exclude_self_match=self.exclude_self_match,
        )
        metric_storage = {
            level: []
            for level in relevance_computer.levels
        }

        category_storage = defaultdict(lambda: defaultdict(list))
        error_cases = []

        category_field = self.get_category_field(metadata)
        for start in range(0, len(query_indices), self.query_batch_size):
            batch_indices = query_indices[start:start + self.query_batch_size]
            query_embeddings = embeddings[batch_indices]

            retrieved_indices_batch, retrieved_scores_batch = searcher.search_batch(
                query_embeddings=query_embeddings,
                all_embeddings=embeddings,
                query_indices=batch_indices,
            )

            for row_number, query_index in enumerate(batch_indices):
                query_row = metadata.iloc[query_index]
                retrieved_indices = retrieved_indices_batch[row_number]
                retrieved_scores = retrieved_scores_batch[row_number]

                if category_field:
                    category = str(query_row.get(category_field, "unknown"))
                else:
                    category = "unknown"

                for level_name, fields in relevance_computer.levels.items():
                    relevance, total_relevant, _ = relevance_computer.get_relevance_list(
                        retrieved_indices=retrieved_indices,
                        fields=fields,
                        query_index=query_index,
                    )

                    metrics = evaluate_ranking(
                        relevance=relevance,
                        total_relevant=total_relevant,
                        k_values=self.k_values,
                    )

                    metric_storage[level_name].append(metrics)
                    category_storage[level_name][category].append(metrics)

                strict_fields = relevance_computer.levels.get("strict", ["articleType"])

                if retrieved_indices:
                    _, _, strict_mask = relevance_computer.get_relevance_list(
                        retrieved_indices=retrieved_indices,
                        fields=strict_fields,
                        query_index=query_index,
                    )

                    top1_index = retrieved_indices[0]
                    top1_score = retrieved_scores[0]

                    if not bool(strict_mask[top1_index]):
                        error_cases.append(
                            build_error_case(
                                query_row=query_row,
                                candidate_row=metadata.iloc[top1_index],
                                score=top1_score,
                                query_index=query_index,
                                candidate_index=top1_index,
                            )
                        )

            logging.info(
                f"Evaluated {min(start + self.query_batch_size, len(query_indices))}/{len(query_indices)} queries"
            )

        metrics_output = self.build_metrics_output(
            metadata=metadata,
            embeddings=embeddings,
            query_indices=query_indices,
            relevance_computer=relevance_computer,
            metric_storage=metric_storage,
        )

        category_rows = self.build_category_rows(category_storage)

        saved_paths = self.report_writer.save(
            metrics_output=metrics_output,
            category_rows=category_rows,
            error_cases=error_cases,
        )

        logging.info(f"Evaluation metrics saved to {saved_paths['metrics_path']}")
        logging.info(f"Category breakdown saved to {saved_paths['category_breakdown_path']}")
        logging.info(f"Error cases saved to {saved_paths['error_cases_path']}")

        return metrics_output