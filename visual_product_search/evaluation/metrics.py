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

def average_precision_at_k(relevance, k, total_relevant=None):
    values = _to_binary_array(relevance)[:k]
    
    if len(values) == 0:
        return 0.0
    
    score = 0.0
    hits = 0.0
    
    for idx, item in enumerate(values, start=1):
        if item > 0:
            hits += 1.0
            score += hits / idx

    if total_relevant is None:
        denominator = hits
    else:
        denominator = min(float(total_relevant), float(k))

    if denominator <= 0:
        return 0.0

    return float(score / denominator)


def dcg_at_k(relevance, k):
    values = _to_binary_array(relevance)[:k]
    
    if len(values) == 0:
        return 0.0
    
    discounts = np.log2(np.arange(2, len(values) + 2))
    return float(np.sum(values / discounts))


def ndcg_at_k(relevance, total_relevant, k):
    if total_relevant <= 0:
        return 0.0

    dcg = dcg_at_k(relevance, k)
    ideal_hits = int(min(total_relevant, k))
    ideal = np.ones(ideal_hits, dtype=np.float32)
    ideal_dcg = dcg_at_k(ideal, ideal_hits)

    if ideal_dcg == 0:
        return 0.0

    return float(dcg / ideal_dcg)


def mrr_at_k(relevance, k):
    values = _to_binary_array(relevance)[:k]

    for idx, item in enumerate(values, start=1):
        if item > 0:
            return float(1.0 / idx)

    return 0.0


def evaluate_ranking(relevance, total_relevant, k_values):
    result = {}

    for k in k_values:
        result[f"precision@{k}"] = precision_at_k(relevance, k)
        result[f"recall@{k}"] = recall_at_k(relevance, total_relevant, k)
        result[f"map@{k}"] = average_precision_at_k(relevance, k, total_relevant)
        result[f"ndcg@{k}"] = ndcg_at_k(relevance, total_relevant, k)
        result[f"mrr@{k}"] = mrr_at_k(relevance, k)

    return result


def average_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}

    keys = metric_dicts[0].keys()

    return {
        key: float(np.mean([item[key] for item in metric_dicts]))
        for key in keys
    }