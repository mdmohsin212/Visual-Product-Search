import numpy as np

class RelevanceComputer:
    def __init__(self, metadata, evaluation_config):
        self.metadata = metadata
        self.evaluation_config = evaluation_config
        self.exclude_self_match = bool(
            evaluation_config.get("exclude_self_match", True)
        )

        self.levels = self.get_relevance_levels()
        self.field_cache = self.build_field_cache()

    def get_relevance_levels(self):
        raw_levels = self.evaluation_config.get("relevance_levels", {})
        levels = {}

        for level_name, value in raw_levels.items():
            fields = value.get("fields", value) if isinstance(value, dict) else value
            fields = [field for field in fields if field]
            if fields:
                levels[level_name] = fields

        if not levels:
            levels = {
                "strict": ["articleType"],
                "medium": ["subCategory"],
                "soft": ["gender", "masterCategory"],
            }

        return levels

    def build_field_cache(self):
        cache = {}
        all_fields = sorted({field for fields in self.levels.values() for field in fields})

        for field in all_fields:
            if field not in self.metadata.columns:
                self.metadata[field] = ""

            cache[field] = (
                self.metadata[field]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .to_numpy()
            )

        return cache

    def get_mask(self, fields, query_index):
        first_field = fields[0]
        mask = np.ones(len(self.field_cache[first_field]), dtype=bool)

        for field in fields:
            values = self.field_cache[field]
            mask &= values == values[query_index]

        if self.exclude_self_match:
            mask[query_index] = False

        return mask

    def get_relevance_list(self, retrieved_indices, fields, query_index):
        mask = self.get_mask(fields, query_index)
        relevance = [bool(mask[index]) for index in retrieved_indices]
        total_relevant = int(mask.sum())

        return relevance, total_relevant, mask