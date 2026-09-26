"""Stage 1: ASR with word-level timestamps, via faster-whisper.

Runs on CPU (CTranslate2 backend has no Apple MPS support), int8 quantized
for speed. Language is passed explicitly from the manifest rather than
auto-detected, since auto-detect on code-switched Hindi/English audio can
misfire. Speaker/title names are passed as an initial_prompt to bias
transcription of proper nouns that Whisper would otherwise mangle.
"""
from __future__ import annotations

import argparse
import sys

from faster_whisper import WhisperModel

from ingest.common import TRANSCRIPTS_DIR, get_logger, get_recording, load_manifest, save_json

log = get_logger("transcribe")

_MODEL = None


def get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        log.info("Loading faster-whisper large-v3 (int8, cpu)...")
        _MODEL = WhisperModel("large-v3", device="cpu", compute_type="int8")
    return _MODEL


def transcribe_recording(recording_id: str, force: bool = False) -> dict:
    rec = get_recording(recording_id)
    out_path = TRANSCRIPTS_DIR / f"{recording_id}.json"
    if out_path.exists() and not force:
        log.info(f"[{recording_id}] cached transcript exists, skipping (use --force to redo)")
        return {}

    model = get_model()
    initial_prompt = f"{rec.title}. Speakers: {', '.join(rec.speakers)}."

    log.info(f"[{recording_id}] transcribing ({rec.language}, {rec.duration_sec:.0f}s)...")
    segments, info = model.transcribe(
        str(rec.audio_abspath),
        language=rec.language,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        vad_filter=True,
    )

    words = []
    full_text_parts = []
    for seg in segments:
        full_text_parts.append(seg.text)
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word.strip(),
                        # Whisper's tokenizer encodes "attaches directly to
                        # previous word" (e.g. "-planetary," after "multi")
                        # as the absence of a leading space - preserve that
                        # signal so chunk.py can join words correctly.
                        "leading_space": w.word.startswith(" "),
                        "start_ms": round(w.start * 1000),
                        "end_ms": round(w.end * 1000),
                        "confidence": round(float(w.probability), 4),
                    }
                )

    result = {
        "recording_id": recording_id,
        "language": rec.language,
        "detected_language": info.language,
        "detected_language_probability": round(float(info.language_probability), 4),
        "full_text": "".join(full_text_parts).strip(),
        "words": words,
    }
    save_json(out_path, result)
    log.info(f"[{recording_id}] done: {len(words)} words -> {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording_id", nargs="?", help="manifest recording id, or omit with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.all:
        for rec in load_manifest():
            transcribe_recording(rec.id, force=args.force)
    elif args.recording_id:
        transcribe_recording(args.recording_id, force=args.force)
    else:
        ap.error("provide a recording_id or --all")


if __name__ == "__main__":
    main()
