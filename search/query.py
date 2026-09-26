"""Hybrid search: lexical (FTS) + fuzzy (trigram, word_similarity) +
semantic (HNSW), fused with Reciprocal Rank Fusion (RRF), reranked with a
cross-encoder, with highlighted snippets and a precise word-level jump
timestamp.

Deferred to a later pass: the query parser (phrases, speaker:/file:
filters), adjacent-hit merging across utterances, and the "only run
trigram for rare/capitalized terms" optimization - here it always runs
all three retrieval legs.
"""
from __future__ import annotations

import argparse
import os
import re

import psycopg
import torch
from pgvector import Vector
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest.common import DATABASE_URL, get_logger

log = get_logger("search")

RRF_K = 60
RERANK_POOL = int(os.environ.get("NUVIX_RERANK_POOL", "30"))  # candidates the cross-encoder scores; lower = faster on CPU
MIN_RERANK_SCORE = 0.006  # applied to semantic-only hits: below this, the cross-encoder is
                          # saying "not actually relevant" - drop rather than pad with noise.
                          # Was 0.05, which dropped correct Hindi/cross-lingual hits the reranker
                          # ranks #1 but scores 0.003-0.02 (e.g. "batsman", "how to handle ego
                          # and pride"). Lowered to just above the highest score seen on the 7
                          # true-negative gold queries (0.0052), 2026-09-26. Tuned on a tiny set:
                          # rerank scores for real hits and negatives overlap below ~0.006, so a
                          # few correct low-scoring hits (q03, q15, q28) are still lost.
LEXICAL_SUPPORTED_MIN_SCORE = 0.0  # applied when lexical or fuzzy already found the hit via
                          # literal/near-literal term matching - that's objective evidence the
                          # cross-encoder's absolute score can't override. Found via manual
                          # testing: reranker scored genuine "Stanford" hits (agreed on by all
                          # 3 retrieval legs) at 0.006-0.007, while an equivalent-style query
                          # ("Tesla") scored 0.166 - the reranker isn't reliably calibrated
                          # across topics, so an absolute cutoff shouldn't override a literal
                          # term match. Kept at 0.0 rather than removed entirely as a floor
                          # against a future degenerate-text edge case slipping through.
_MODEL = None
_RERANKER = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info(f"Loading BAAI/bge-m3 on {device}...")
        _MODEL = SentenceTransformer("BAAI/bge-m3", device=device)
    return _MODEL


def get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info(f"Loading BAAI/bge-reranker-v2-m3 on {device}...")
        _RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
    return _RERANKER


def lexical_search(cur, query: str, limit: int = 50) -> list[tuple]:
    cur.execute(
        """
        SELECT u.id, ts_rank_cd(u.tsv_simple, websearch_to_tsquery('simple', %s)) AS score
        FROM utterances u
        WHERE u.tsv_simple @@ websearch_to_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, query, limit),
    )
    return cur.fetchall()


def fuzzy_search(cur, query: str, limit: int = 20) -> list[tuple]:
    """Uses word_similarity (best-matching substring), not similarity()
    (whole-string overlap) - similarity() badly underscores a short query
    against a long utterance even when a strong local match exists (e.g.
    "ultracapacitors" vs "...advanced ultra capacitors..." inside a much
    longer sentence). Same GIN trigram index accelerates both operators."""
    cur.execute(
        """
        SELECT u.id, word_similarity(%s, u.text_norm) AS score
        FROM utterances u
        WHERE %s <%% u.text_norm
        ORDER BY score DESC
        LIMIT %s
        """,
        (query.lower(), query.lower(), limit),
    )
    return cur.fetchall()


def semantic_search(cur, query_vector, limit: int = 50) -> list[tuple]:
    cur.execute(
        """
        SELECT u.id, 1 - (u.embedding <=> %s) AS score
        FROM utterances u
        WHERE u.embedding IS NOT NULL
        ORDER BY u.embedding <=> %s
        LIMIT %s
        """,
        (query_vector, query_vector, limit),
    )
    return cur.fetchall()


def rrf_fuse(*ranked_lists: list[tuple], k: int = RRF_K) -> dict[int, dict]:
    """Each ranked_lists entry is [(utterance_id, raw_score), ...] in rank order.
    Returns {utterance_id: {"rrf_score": float, "hits": ["lexical", ...]}}."""
    labels = ["lexical", "fuzzy", "semantic"]
    fused: dict[int, dict] = {}
    for label, ranked in zip(labels, ranked_lists):
        for rank, (uid, raw_score) in enumerate(ranked, start=1):
            entry = fused.setdefault(uid, {"rrf_score": 0.0, "hits": [], "raw_scores": {}})
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["hits"].append(label)
            entry["raw_scores"][label] = round(float(raw_score), 4)
    return fused


def _query_terms(query: str) -> list[str]:
    return re.findall(r"\w+", query.lower())


def _best_word_timestamp(cur, utterance_id: int, terms: list[str]) -> int | None:
    """Find the earliest word in this utterance matching one of the query's
    terms, for a precise jump-to timestamp - falls back to the utterance's
    own start_ms (handled by the caller) for a pure-semantic hit with no
    literal term overlap."""
    if not terms:
        return None
    cur.execute(
        """
        SELECT start_ms FROM words
        WHERE utterance_id = %s AND lower(regexp_replace(word, '[^\\w]', '', 'g')) = ANY(%s)
        ORDER BY start_ms
        LIMIT 1
        """,
        (utterance_id, terms),
    )
    row = cur.fetchone()
    return row[0] if row else None


ALL_LEGS = frozenset({"lexical", "fuzzy", "semantic"})


def search(
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    min_rerank_score: float = MIN_RERANK_SCORE,
    legs: frozenset[str] = ALL_LEGS,
) -> list[dict]:
    """`legs` restricts which retrieval methods actually run - used by the
    ablation study (eval/run_eval.py) to compare lexical-only, semantic-
    only, and full-hybrid recall against the gold query set."""
    model = get_model()
    query_vector = Vector(model.encode(query, normalize_embeddings=True))
    terms = _query_terms(query)

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            lex = lexical_search(cur, query) if "lexical" in legs else []
            fuz = fuzzy_search(cur, query) if "fuzzy" in legs else []
            sem = semantic_search(cur, query_vector) if "semantic" in legs else []

            fused = rrf_fuse(lex, fuz, sem)
            pool_size = RERANK_POOL if rerank else top_k
            pool_ids = sorted(fused, key=lambda uid: fused[uid]["rrf_score"], reverse=True)[:pool_size]

            if not pool_ids:
                return []

            cur.execute(
                """
                SELECT u.id, r.title, r.id AS recording_id, s.display_name,
                       u.start_ms, u.end_ms, u.text_raw,
                       ts_headline('simple', u.text_raw, websearch_to_tsquery('simple', %s),
                                   'StartSel=**, StopSel=**, HighlightAll=true') AS snippet
                FROM utterances u
                JOIN recordings r ON u.recording_id = r.id
                JOIN speakers s ON u.speaker_id = s.id
                WHERE u.id = ANY(%s)
                """,
                (query, pool_ids),
            )
            rows = {r[0]: r for r in cur.fetchall()}

            word_ts = {uid: _best_word_timestamp(cur, uid, terms) for uid in pool_ids}

    candidates = [uid for uid in pool_ids if uid in rows]
    if not candidates:
        return []

    if rerank:
        reranker = get_reranker()
        pairs = [(query, rows[uid][6]) for uid in candidates]
        scores = reranker.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        rerank_scores = {candidates[i]: round(float(scores[i]), 4) for i in range(len(candidates))}

        def _clears_threshold(uid: int) -> bool:
            has_literal_support = "lexical" in fused[uid]["hits"] or "fuzzy" in fused[uid]["hits"]
            threshold = LEXICAL_SUPPORTED_MIN_SCORE if has_literal_support else min_rerank_score
            return rerank_scores[uid] >= threshold

        # Drop below-threshold results instead of padding up to top_k with
        # noise the cross-encoder itself is saying is irrelevant - but don't
        # let the reranker's absolute score override a literal term match
        # lexical/fuzzy already found (see LEXICAL_SUPPORTED_MIN_SCORE above).
        ranked_ids = [candidates[i] for i in order if _clears_threshold(candidates[i])][:top_k]
    else:
        ranked_ids = candidates[:top_k]
        rerank_scores = {}

    results = []
    for uid in ranked_ids:
        _, title, recording_id, speaker, start_ms, end_ms, text, snippet = rows[uid]
        jump_ms = word_ts.get(uid) or start_ms
        mm, ss = divmod(jump_ms // 1000, 60)
        results.append(
            {
                "utterance_id": uid,
                "rrf_score": round(fused[uid]["rrf_score"], 5),
                "rerank_score": rerank_scores.get(uid),
                "hit_methods": fused[uid]["hits"],
                "raw_scores": fused[uid]["raw_scores"],
                "recording_id": recording_id,
                "title": title,
                "speaker": speaker,
                "start_ms": start_ms,  # utterance's own bounds - used for gold-interval matching in eval/
                "end_ms": end_ms,
                "timestamp": f"{mm:02d}:{ss:02d}",
                "jump_ms": jump_ms,  # word-level position of the match - where a player should seek to
                "text": text,
                "snippet": snippet,
            }
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--min-score", type=float, default=MIN_RERANK_SCORE)
    args = ap.parse_args()

    results = search(args.query, top_k=args.k, rerank=not args.no_rerank, min_rerank_score=args.min_score)
    if not results:
        print(f"No results above min-score={args.min_score} (irrelevant hits were dropped, not padded to --k).")
        return
    if len(results) < args.k and not args.no_rerank:
        print(f"(only {len(results)}/{args.k} results cleared min-score={args.min_score})\n")

    for i, r in enumerate(results, 1):
        methods = "+".join(sorted(set(r["hit_methods"])))
        rerank_str = f", rerank={r['rerank_score']:.3f}" if r["rerank_score"] is not None else ""
        print(
            f"{i:2d}. [rrf={r['rrf_score']:.4f}{rerank_str} | {methods:20s}] "
            f"{r['speaker']:15s} {r['timestamp']}  ({r['recording_id']})"
        )
        print(f"      {r['snippet']}")
        print()


if __name__ == "__main__":
    main()
