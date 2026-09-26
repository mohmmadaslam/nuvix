"""Unit tests for eval/metrics.py - pure logic, no Postgres/models needed.
Fast regression guard for the retrieval-quality math itself, independent
of the full pipeline (see tests/test_recall.py for the end-to-end version).
"""
from __future__ import annotations

import pytest

from eval.metrics import intervals_overlap, ndcg_at_k, recall_at_k, reciprocal_rank

REC = "some-recording"


def _res(start_ms, end_ms=None):
    return {"recording_id": REC, "start_ms": start_ms, "end_ms": end_ms or start_ms + 10_000}


def _rel(start_ms, end_ms=None):
    return {"recording_id": REC, "start_ms": start_ms, "end_ms": end_ms or start_ms + 10_000}


def test_intervals_overlap_direct():
    assert intervals_overlap(1000, 5000, 3000, 7000)  # genuine overlap
    assert not intervals_overlap(1000, 2000, 50000, 60000)  # far apart


def test_intervals_overlap_tolerance():
    # 3s apart, within default 3000ms tolerance
    assert intervals_overlap(1000, 2000, 5000, 6000, tolerance_ms=3000)
    # well outside tolerance
    assert not intervals_overlap(1000, 2000, 20000, 21000, tolerance_ms=3000)


def test_recall_at_k_perfect_hit():
    results = [_res(1000)]
    relevant = [_rel(1000)]
    assert recall_at_k(results, relevant, k=5) == 1.0


def test_recall_at_k_miss():
    results = [_res(999_000)]
    relevant = [_rel(1000)]
    assert recall_at_k(results, relevant, k=5) == 0.0


def test_recall_at_k_partial_coverage_of_multiple_relevant():
    relevant = [_rel(1000), _rel(500_000)]
    results = [_res(1000)]  # only covers the first of two relevant intervals
    assert recall_at_k(results, relevant, k=5) == 0.5


def test_recall_at_k_negative_query_no_results_is_correct():
    assert recall_at_k([], [], k=5) == 1.0


def test_recall_at_k_negative_query_false_positive_is_wrong():
    assert recall_at_k([_res(1000)], [], k=5) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([_res(1000)], [_rel(1000)]) == 1.0


def test_reciprocal_rank_third_position():
    results = [_res(999_000), _res(998_000), _res(1000)]
    assert reciprocal_rank(results, [_rel(1000)]) == pytest.approx(1 / 3)


def test_ndcg_bounded_even_with_duplicate_interval_matches():
    """Regression test for a real bug found 2026-09-26: multiple returned
    results overlapping the SAME gold interval must not each add
    independent credit to dcg, or nDCG can exceed 1.0 (mathematically
    impossible for a correct implementation). Here 3 results all overlap
    the single relevant interval - nDCG must still be <= 1.0."""
    relevant = [_rel(1000, 11000)]
    results = [_res(1500, 11500), _res(2000, 12000), _res(2500, 12500)]
    score = ndcg_at_k(results, relevant, k=10)
    assert 0.0 <= score <= 1.0


def test_ndcg_perfect_ranking_is_one():
    relevant = [_rel(1000)]
    results = [_res(1000)]
    assert ndcg_at_k(results, relevant, k=10) == 1.0


def test_ndcg_negative_query():
    assert ndcg_at_k([], [], k=10) == 1.0
    assert ndcg_at_k([_res(1000)], [], k=10) == 0.0
