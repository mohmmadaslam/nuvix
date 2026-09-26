"""Runs the gold query set (data/gold/queries.yaml) through search across
several retrieval configurations and reports recall@1/5/10, MRR, and
nDCG@10 - overall and per query category, per docs/ARCHITECTURE.md
Section 7.

Configurations compared (the ablation study):
  lexical_only   - lexical leg alone, no rerank
  fuzzy_only     - fuzzy leg alone, no rerank
  semantic_only  - semantic leg alone, no rerank
  hybrid         - all 3 legs + RRF fusion, no rerank
  hybrid_rerank  - all 3 legs + RRF fusion + cross-encoder rerank (the
                   system's actual default configuration)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from ingest.common import ROOT, get_logger
from search.query import search

log = get_logger("eval")

GOLD_PATH = ROOT / "data" / "gold" / "queries.yaml"

CONFIGS = {
    "lexical_only": dict(legs=frozenset({"lexical"}), rerank=False),
    "fuzzy_only": dict(legs=frozenset({"fuzzy"}), rerank=False),
    "semantic_only": dict(legs=frozenset({"semantic"}), rerank=False),
    "hybrid": dict(legs=frozenset({"lexical", "fuzzy", "semantic"}), rerank=False),
    "hybrid_rerank": dict(legs=frozenset({"lexical", "fuzzy", "semantic"}), rerank=True),
}

K_VALUES = (1, 5, 10)


def load_gold() -> list[dict]:
    with open(GOLD_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["queries"]


def evaluate_config(gold: list[dict], config_name: str, config: dict, top_k: int = 10) -> dict:
    per_query = []
    for q in gold:
        results = search(q["query"], top_k=top_k, **config)
        row = {"id": q["id"], "category": q["category"], "query": q["query"]}
        for k in K_VALUES:
            row[f"recall@{k}"] = recall_at_k(results, q["relevant"], k)
        row["mrr"] = reciprocal_rank(results, q["relevant"])
        row["ndcg@10"] = ndcg_at_k(results, q["relevant"], 10)
        per_query.append(row)

    n = len(per_query)
    summary = {"config": config_name, "n_queries": n}
    for k in K_VALUES:
        summary[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_query) / n
    summary["mrr"] = sum(r["mrr"] for r in per_query) / n
    summary["ndcg@10"] = sum(r["ndcg@10"] for r in per_query) / n

    by_category = defaultdict(list)
    for r in per_query:
        by_category[r["category"]].append(r)
    category_summary = {}
    for cat, rows in by_category.items():
        cn = len(rows)
        category_summary[cat] = {
            "n": cn,
            **{f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / cn for k in K_VALUES},
            "mrr": sum(r["mrr"] for r in rows) / cn,
        }

    return {"summary": summary, "by_category": category_summary, "per_query": per_query}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS) + ["all"], default="all")
    ap.add_argument("--out", type=str, default=None, help="write full JSON report to this path")
    args = ap.parse_args()

    gold = load_gold()
    log.info(f"Loaded {len(gold)} gold queries from {GOLD_PATH}")

    configs_to_run = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    all_reports = {}

    for name, cfg in configs_to_run.items():
        log.info(f"Running config: {name}...")
        report = evaluate_config(gold, name, cfg)
        all_reports[name] = report

    print("\n=== Summary (overall) ===")
    header = f"{'config':16s} " + " ".join(f"{'recall@'+str(k):>10s}" for k in K_VALUES) + f" {'mrr':>8s} {'ndcg@10':>8s}"
    print(header)
    for name, report in all_reports.items():
        s = report["summary"]
        row = f"{name:16s} " + " ".join(f"{s[f'recall@{k}']:>10.3f}" for k in K_VALUES) + f" {s['mrr']:>8.3f} {s['ndcg@10']:>8.3f}"
        print(row)

    print("\n=== By category (hybrid_rerank, if run) ===")
    if "hybrid_rerank" in all_reports:
        cats = all_reports["hybrid_rerank"]["by_category"]
        for cat, s in sorted(cats.items()):
            print(
                f"  {cat:25s} n={s['n']:2d}  recall@1={s['recall@1']:.2f}  "
                f"recall@5={s['recall@5']:.2f}  recall@10={s['recall@10']:.2f}  mrr={s['mrr']:.2f}"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(all_reports, indent=2))
        log.info(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
