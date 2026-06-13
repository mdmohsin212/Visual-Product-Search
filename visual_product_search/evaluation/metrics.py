import numpy as np

def _to_binary_array(relevance):
    return np.asarray(relevance, dtype=np.float32)

def precision_at_k(relevance, k):
    values = _to_binary_array(relevance)[:k]
    
    if len(values) == 0:
        return 0.0
    
    return float(np.sum(values) / len(values))

def recall_at_k(relevance, total_relevant, k):
    if total_relevant <= 0:
        return 0.0

    values = _to_binary_array(relevance)[:k]
    return float(np.sum(values) / total_relevant)