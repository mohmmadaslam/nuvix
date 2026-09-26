"""Retrieval metrics for the gold query set: recall@k, MRR, nDCG@k.

Matching is by time-interval overlap (with tolerance), not chunk/utterance
ID equality - per docs/ARCHITECTURE.md Section 7, this keeps gold labels
valid even if chunking, the embedding model, or FTS config changes later.
"""
from __future__ import annotations

import math

TOLERANCE_MS = 3000


def intervals_overlap(
    a_start: int, a_end: int, b_start: int, b_end: int, tolerance_ms: int = TOLERANCE_MS
) -> bool:
    return not (a_end + tolerance_ms < b_start or a_start - tolerance_ms > b_end)


def is_hit(result: dict, relevant: list[dict]) -> bool:
    for r in relevant:
        if result["recording_id"] != r["recording_id"]:
            continue
        if intervals_overlap(result["start_ms"], result["end_ms"], r["start_ms"], r["end_ms"]):
            return True
    return False


def recall_at_k(results: list[dict], relevant: list[dict], k: int) -> float:
    """Fraction of DISTINCT relevant intervals covered by at least one hit
    in the top-k results. For a negative query (relevant=[]): 1.0 if the
    top-k is empty (correctly returned nothing), else 0.0 (a false
    positive on a query with no true answer)."""
    if not relevant:
        return 1.0 if not results[:k] else 0.0
    covered = set()
    for res in results[:k]:
        for j, r in enumerate(relevant):
            if res["recording_id"] == r["recording_id"] and intervals_overlap(
                res["start_ms"], res["end_ms"], r["start_ms"], r["end_ms"]
            ):
                covered.add(j)
    return len(covered) / len(relevant)


def reciprocal_rank(results: list[dict], relevant: list[dict]) -> float:
    if not relevant:
        return 1.0 if not results else 0.0
    for i, res in enumerate(results, start=1):
        if is_hit(res, relevant):
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: list[dict], relevant: list[dict], k: int) -> float:
    """Binary relevance nDCG. Credits each distinct relevant interval at
    most once (like recall_at_k's `covered` set) - without this, two
    returned results both overlapping the same gold interval would each
    add to dcg independently, letting dcg exceed idcg (nDCG > 1.0, which
    is mathematically impossible for a correct implementation)."""
    if not relevant:
        return 1.0 if not results[:k] else 0.0
    dcg = 0.0
    covered: set[int] = set()
    for i, res in enumerate(results[:k], start=1):
        for j, r in enumerate(relevant):
            if j in covered:
                continue
            if res["recording_id"] == r["recording_id"] and intervals_overlap(
                res["start_ms"], res["end_ms"], r["start_ms"], r["end_ms"]
            ):
                covered.add(j)
                dcg += 1.0 / math.log2(i + 1)
                break
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
