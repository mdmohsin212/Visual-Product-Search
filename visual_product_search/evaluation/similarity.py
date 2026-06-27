import numpy as np

class SimilaritySearcher:
    def __init__(self, top_k, exclude_self_match=True):
        self.top_k = int(top_k)
        self.exclude_self_match = bool(exclude_self_match)

    def search_batch(self, query_embeddings, all_embeddings, query_indices):
        scores = query_embeddings @ all_embeddings.T
        if self.exclude_self_match:
            for local_index, global_index in enumerate(query_indices):
                scores[local_index, global_index] = -np.inf

        candidate_count = min(self.top_k + 1, all_embeddings.shape[0])
        unordered_indices = np.argpartition(
            -scores,
            kth=candidate_count - 1,
            axis=1,
        )[:, :candidate_count]

        unordered_scores = np.take_along_axis(scores, unordered_indices, axis=1)
        order = np.argsort(-unordered_scores, axis=1)

        ordered_indices = np.take_along_axis(unordered_indices, order, axis=1)
        ordered_scores = np.take_along_axis(unordered_scores, order, axis=1)

        batch_indices = []
        batch_scores = []

        for row_id, query_index in enumerate(query_indices):
            result_indices = []
            result_scores = []

            for index, score in zip(ordered_indices[row_id], ordered_scores[row_id]):
                if self.exclude_self_match and index == query_index:
                    continue

                if len(result_indices) == self.top_k:
                    break

                result_indices.append(int(index))
                result_scores.append(float(score))

            batch_indices.append(result_indices)
            batch_scores.append(result_scores)

        return batch_indices, batch_scores