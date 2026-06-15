import pytest
from visual_product_search.evaluation.metrics import (
    average_precision_at_k,
    dcg_at_k,
    evaluate_ranking,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    relevance = [1, 0, 1, 1, 0]
    assert precision_at_k(relevance, 5) == pytest.approx(0.6)


def test_recall_at_k():
    relevance = [1, 0, 1, 1, 0]
    assert recall_at_k(relevance, total_relevant=6, k=5) == pytest.approx(0.5)


def test_average_precision_at_k():
    relevance = [1, 0, 1, 1, 0]
    value = average_precision_at_k(relevance, k=5, total_relevant=3)
    assert value == pytest.approx((1 / 1 + 2 / 3 + 3 / 4) / 3)


def test_dcg_at_k():
    relevance = [1, 0, 1]
    value = dcg_at_k(relevance, 3)
    expected = 1 / 1 + 0 / 1.584962500721156 + 1 / 2
    assert value == pytest.approx(expected)


def test_ndcg_at_k():
    relevance = [1, 0, 1]
    value = ndcg_at_k(relevance, total_relevant=2, k=3)
    assert 0 <= value <= 1


def test_mrr_at_k():
    relevance = [0, 0, 1, 0]
    assert mrr_at_k(relevance, 4) == pytest.approx(1 / 3)


def test_evaluate_ranking():
    relevance = [1, 0, 1, 0, 1]

    result = evaluate_ranking(relevance, total_relevant=5, k_values=[1, 5])

    assert "precision@1" in result
    assert "recall@5" in result
    assert "map@5" in result
    assert "ndcg@5" in result
    assert "mrr@5" in result