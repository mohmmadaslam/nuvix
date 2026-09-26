"""Stage 5: dense embeddings for each utterance's context_text, via bge-m3.

bge-m3 is multilingual (covers the Hindi/Hinglish files as well as the
English one) and runs locally via sentence-transformers, MPS-accelerated.
Embeddings are L2-normalized so cosine similarity == dot product.
"""
from __future__ import annotations

import argparse

import torch
from sentence_transformers import SentenceTransformer

from ingest.common import (
    EMBEDDED_DIR,
    UTTERANCES_DIR,
    get_logger,
    load_json,
    load_manifest,
    save_json,
)

log = get_logger("embed")

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info(f"Loading {EMBEDDING_MODEL_NAME} on {device}...")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _MODEL


def embed_recording(recording_id: str, force: bool = False) -> dict:
    out_path = EMBEDDED_DIR / f"{recording_id}.json"
    if out_path.exists() and not force:
        log.info(f"[{recording_id}] cached embeddings exist, skipping")
        return {}

    data = load_json(UTTERANCES_DIR / f"{recording_id}.json")
    utterances = data["utterances"]
    if not utterances:
        log.warning(f"[{recording_id}] no utterances to embed")
        save_json(out_path, data)
        return data

    model = get_model()
    texts = [u["context_text"] for u in utterances]
    log.info(f"[{recording_id}] embedding {len(texts)} utterances...")
    vectors = model.encode(
        texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
    )

    for u, v in zip(utterances, vectors):
        u["embedding"] = v.tolist()
        u["embedding_model"] = EMBEDDING_MODEL_NAME

    save_json(out_path, data)
    log.info(f"[{recording_id}] done: {len(utterances)} embeddings (dim={len(vectors[0])}) -> {out_path}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording_id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.all:
        for rec in load_manifest():
            embed_recording(rec.id, force=args.force)
    elif args.recording_id:
        embed_recording(args.recording_id, force=args.force)
    else:
        ap.error("provide a recording_id or --all")


if __name__ == "__main__":
    main()
