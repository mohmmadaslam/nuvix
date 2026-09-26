"""Stage 3: assign each transcribed word to a speaker, using diarization
segments, and map diarization labels (SPEAKER_00/01) to real names from
the manifest via first-appearance order.
"""
from __future__ import annotations

import argparse

from ingest.common import (
    DIARIZATION_DIR,
    SPEAKER_WORDS_DIR,
    TRANSCRIPTS_DIR,
    get_logger,
    get_recording,
    load_json,
    load_manifest,
    save_json,
)

log = get_logger("assign_speakers")


def _overlap_ms(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def assign_word_speakers(words: list[dict], segments: list[dict]) -> list[str | None]:
    """For each word, return the diarization label with max time-overlap.
    Falls back to nearest segment by midpoint distance if there's no overlap
    (e.g. a word falls in a diarization gap)."""
    labels = []
    for w in words:
        best_label, best_overlap = None, 0
        for s in segments:
            ov = _overlap_ms(w["start_ms"], w["end_ms"], s["start_ms"], s["end_ms"])
            if ov > best_overlap:
                best_overlap, best_label = ov, s["speaker_label"]
        if best_label is None and segments:
            mid = (w["start_ms"] + w["end_ms"]) / 2
            best_label = min(
                segments, key=lambda s: abs((s["start_ms"] + s["end_ms"]) / 2 - mid)
            )["speaker_label"]
        labels.append(best_label)
    return labels


def smooth_labels(labels: list[str | None]) -> list[str | None]:
    """Reassign isolated single-word label flips to match both neighbors."""
    smoothed = list(labels)
    for i in range(1, len(smoothed) - 1):
        prev_l, cur_l, next_l = smoothed[i - 1], smoothed[i], smoothed[i + 1]
        if prev_l is not None and prev_l == next_l and cur_l != prev_l:
            smoothed[i] = prev_l
    return smoothed


def build_speaker_map(labels: list[str | None], words: list[dict], speaker_names: list[str]) -> dict:
    """Map diarization labels to manifest speaker names by first-appearance order."""
    first_seen_order = []
    for label, w in zip(labels, words):
        if label is not None and label not in first_seen_order:
            first_seen_order.append(label)
    mapping = {}
    for i, label in enumerate(first_seen_order):
        mapping[label] = speaker_names[i] if i < len(speaker_names) else f"Unknown ({label})"
    return mapping


def assign_speakers(recording_id: str, force: bool = False) -> dict:
    out_path = SPEAKER_WORDS_DIR / f"{recording_id}.json"
    if out_path.exists() and not force:
        log.info(f"[{recording_id}] cached speaker assignment exists, skipping")
        return {}

    rec = get_recording(recording_id)
    transcript = load_json(TRANSCRIPTS_DIR / f"{recording_id}.json")
    diarization = load_json(DIARIZATION_DIR / f"{recording_id}.json")

    words = transcript["words"]
    segments = diarization["segments"]

    raw_labels = assign_word_speakers(words, segments)
    labels = smooth_labels(raw_labels)
    speaker_map = build_speaker_map(labels, words, rec.speakers)

    flips_smoothed = sum(1 for a, b in zip(raw_labels, labels) if a != b)

    out_words = []
    for w, label in zip(words, labels):
        out_words.append({**w, "speaker_label": label, "speaker": speaker_map.get(label)})

    result = {
        "recording_id": recording_id,
        "speaker_map": speaker_map,
        "words": out_words,
    }
    save_json(out_path, result)
    log.info(
        f"[{recording_id}] done: {len(out_words)} words, speaker_map={speaker_map}, "
        f"smoothed {flips_smoothed} single-word flips -> {out_path}"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording_id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.all:
        for rec in load_manifest():
            assign_speakers(rec.id, force=args.force)
    elif args.recording_id:
        assign_speakers(args.recording_id, force=args.force)
    else:
        ap.error("provide a recording_id or --all")


if __name__ == "__main__":
    main()
