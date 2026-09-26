"""Orchestrates the full ingestion pipeline for one recording (or --all):
transcribe -> diarize -> assign_speakers -> chunk -> embed -> load.

Each stage caches its output to data/pipeline/<stage>/<id>.json and is
skipped on a re-run unless --force is passed, so a failure partway
through doesn't waste already-completed (expensive) work.
"""
from __future__ import annotations

import argparse
import time

from ingest.assign_speakers import assign_speakers
from ingest.chunk import chunk_recording
from ingest.common import get_logger, load_manifest
from ingest.diarize import diarize_recording
from ingest.embed import embed_recording
from ingest.load import load_recording
from ingest.transcribe import transcribe_recording

log = get_logger("run_pipeline")


def run_one(recording_id: str, force: bool = False) -> None:
    t0 = time.time()
    log.info(f"=== [{recording_id}] pipeline start ===")

    transcribe_recording(recording_id, force=force)
    diarize_recording(recording_id, force=force)
    assign_speakers(recording_id, force=force)
    chunk_recording(recording_id, force=force)
    embed_recording(recording_id, force=force)
    load_recording(recording_id)

    log.info(f"=== [{recording_id}] pipeline done in {time.time() - t0:.0f}s ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording_id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore caches, redo every stage")
    args = ap.parse_args()

    if args.all:
        for rec in load_manifest():
            run_one(rec.id, force=args.force)
    elif args.recording_id:
        run_one(args.recording_id, force=args.force)
    else:
        ap.error("provide a recording_id or --all")


if __name__ == "__main__":
    main()
