"""Stage 4: turn -> utterance -> context-window chunking.

Turns: consecutive words from the same speaker.
Utterances (the retrieval unit): a turn split at sentence boundaries into
~10-40s / <=~120-word pieces; turns shorter than that stay whole.
Context window: previous + current + next utterance's normalized text,
used only as the embedding input - the utterance's own timestamp/speaker
is still what gets displayed.

Known simplification (documented limitation, not attempted here): number
normalization in text_norm (e.g. "twenty three" vs "23") is not applied.
Disfluency stripping only covers a conservative English filler list;
Hindi-specific fillers are left as-is rather than guessed at.
"""
from __future__ import annotations

import argparse
import re

from ingest.common import (
    SPEAKER_WORDS_DIR,
    UTTERANCES_DIR,
    get_logger,
    get_recording,
    load_json,
    load_manifest,
    save_json,
)

log = get_logger("chunk")

MIN_UTTERANCE_MS = 10_000
MAX_UTTERANCE_MS = 40_000
MAX_WORDS = 120
SENTENCE_END_RE = re.compile(r"[.?!]$")
FILLER_WORDS = {"um", "uh", "umm", "uhh", "uhm", "mm", "hmm", "erm"}


def filter_degenerate_words(words: list[dict]) -> list[dict]:
    """Drop zero-duration words - a known Whisper hallucination artifact
    (near-silent/hallucinated content, typically low-confidence, sometimes
    appearing near a file's end). A handful of these were observed in the
    elon-musk-build-the-future transcript (identical start_ms==end_ms,
    confidences 0.02-0.79) and would otherwise violate the utterances
    table's CHECK (end_ms > start_ms) constraint."""
    kept = [w for w in words if w["end_ms"] > w["start_ms"]]
    dropped = len(words) - len(kept)
    if dropped:
        log.warning(f"dropped {dropped} zero-duration word(s) (likely ASR hallucination artifacts)")
    return kept


def group_into_turns(words: list[dict]) -> list[dict]:
    words = filter_degenerate_words(words)
    turns = []
    for w in words:
        if not turns or turns[-1]["speaker"] != w["speaker"]:
            turns.append({"speaker": w["speaker"], "words": []})
        turns[-1]["words"].append(w)
    for t in turns:
        t["start_ms"] = t["words"][0]["start_ms"]
        t["end_ms"] = t["words"][-1]["end_ms"]
    return turns


def split_turn_into_utterances(turn: dict) -> list[list[dict]]:
    duration = turn["end_ms"] - turn["start_ms"]
    words = turn["words"]
    if duration <= MAX_UTTERANCE_MS and len(words) <= MAX_WORDS:
        return [words]  # short/medium turn stays whole, per design

    chunks = []
    current: list[dict] = []
    for w in words:
        current.append(w)
        cur_duration = current[-1]["end_ms"] - current[0]["start_ms"]
        is_sentence_end = bool(SENTENCE_END_RE.search(w["word"]))
        long_enough = cur_duration >= MIN_UTTERANCE_MS
        too_long = cur_duration >= MAX_UTTERANCE_MS or len(current) >= MAX_WORDS
        if (is_sentence_end and long_enough) or too_long:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def join_words(words: list[dict]) -> str:
    """Join word tokens respecting Whisper's leading-space signal, so a word
    that attaches directly to the previous one (e.g. "-planetary," after
    "multi") doesn't get a stray space inserted before it. Falls back to a
    hyphen/apostrophe heuristic for transcripts cached before this field
    existed (no leading_space key present)."""
    parts = []
    for i, w in enumerate(words):
        text = w["word"]
        has_leading_space = w.get("leading_space")
        if has_leading_space is None:
            has_leading_space = not text.startswith(("-", "'"))
        if i > 0 and has_leading_space:
            parts.append(" ")
        parts.append(text)
    return "".join(parts)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split(" ") if t.strip(".,!?") not in FILLER_WORDS]
    return " ".join(tokens)


def build_utterances(recording_id: str) -> list[dict]:
    data = load_json(SPEAKER_WORDS_DIR / f"{recording_id}.json")
    turns = group_into_turns(data["words"])

    raw_utterances = []
    for turn in turns:
        for word_group in split_turn_into_utterances(turn):
            raw_utterances.append({"speaker": turn["speaker"], "words": word_group})

    utterances = []
    for seq, u in enumerate(raw_utterances):
        words = u["words"]
        text_raw = join_words(words)
        text_norm = normalize_text(text_raw)
        utterances.append(
            {
                "seq": seq,
                "speaker": u["speaker"],
                "start_ms": words[0]["start_ms"],
                "end_ms": words[-1]["end_ms"],
                "text_raw": text_raw,
                "text_norm": text_norm,
                "words": words,
            }
        )

    # context_text: neighbor text_norm regardless of speaker, embedding input only
    for i, u in enumerate(utterances):
        parts = []
        if i > 0:
            parts.append(utterances[i - 1]["text_norm"])
        parts.append(u["text_norm"])
        if i < len(utterances) - 1:
            parts.append(utterances[i + 1]["text_norm"])
        u["context_text"] = " ".join(p for p in parts if p)

    return utterances


def chunk_recording(recording_id: str, force: bool = False) -> dict:
    out_path = UTTERANCES_DIR / f"{recording_id}.json"
    if out_path.exists() and not force:
        log.info(f"[{recording_id}] cached utterances exist, skipping")
        return {}

    utterances = build_utterances(recording_id)
    result = {"recording_id": recording_id, "utterances": utterances}
    save_json(out_path, result)

    durations = [(u["end_ms"] - u["start_ms"]) / 1000 for u in utterances]
    log.info(
        f"[{recording_id}] done: {len(utterances)} utterances, "
        f"avg {sum(durations)/max(len(durations),1):.1f}s, "
        f"min {min(durations, default=0):.1f}s, max {max(durations, default=0):.1f}s -> {out_path}"
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
            chunk_recording(rec.id, force=args.force)
    elif args.recording_id:
        chunk_recording(args.recording_id, force=args.force)
    else:
        ap.error("provide a recording_id or --all")


if __name__ == "__main__":
    main()
