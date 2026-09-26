"""Stage 2: speaker diarization via pyannote, num_speakers pinned to 2.

pyannote is plain PyTorch, so it gets MPS acceleration on Apple Silicon
where available. num_speakers is passed explicitly (not left for the
pipeline to estimate) since every recording in the golden dataset is
confirmed to have exactly two speakers - this materially improves
diarization accuracy over letting it guess.
"""
from __future__ import annotations

import argparse

import torch
from pyannote.audio import Pipeline

from ingest.common import (
    DIARIZATION_DIR,
    HF_TOKEN,
    get_logger,
    get_recording,
    load_manifest,
    save_json,
)

log = get_logger("diarize")

_PIPELINE = None


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set in .env - required for gated pyannote model")
        log.info("Loading pyannote/speaker-diarization-3.1...")
        _PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN
        )
        if torch.backends.mps.is_available():
            log.info("Moving diarization pipeline to MPS")
            _PIPELINE.to(torch.device("mps"))
    return _PIPELINE


def diarize_recording(recording_id: str, force: bool = False) -> dict:
    rec = get_recording(recording_id)
    out_path = DIARIZATION_DIR / f"{recording_id}.json"
    if out_path.exists() and not force:
        log.info(f"[{recording_id}] cached diarization exists, skipping (use --force to redo)")
        return {}

    pipeline = get_pipeline()
    log.info(f"[{recording_id}] diarizing (num_speakers={rec.num_speakers})...")
    diarization = pipeline(str(rec.audio_abspath), num_speakers=rec.num_speakers)

    segments = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start_ms": round(turn.start * 1000),
                "end_ms": round(turn.end * 1000),
                "speaker_label": speaker,
            }
        )

    result = {"recording_id": recording_id, "segments": segments}
    save_json(out_path, result)
    labels = sorted({s["speaker_label"] for s in segments})
    log.info(f"[{recording_id}] done: {len(segments)} segments, labels={labels} -> {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording_id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.all:
        for rec in load_manifest():
            diarize_recording(rec.id, force=args.force)
    elif args.recording_id:
        diarize_recording(args.recording_id, force=args.force)
    else:
        ap.error("provide a recording_id or --all")


if __name__ == "__main__":
    main()
