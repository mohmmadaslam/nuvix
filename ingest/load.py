"""Stage 6: load a recording's manifest metadata + embedded utterances into Postgres.

Idempotent: deletes any existing rows for the recording_id first (FK
cascades clean up speakers/utterances/words), then inserts fresh.
"""
from __future__ import annotations

import argparse

import psycopg
from pgvector.psycopg import register_vector

from ingest.common import (
    DATABASE_URL,
    EMBEDDED_DIR,
    get_logger,
    get_recording,
    load_json,
    load_manifest,
)

log = get_logger("load")


def load_recording(recording_id: str) -> None:
    rec = get_recording(recording_id)
    data = load_json(EMBEDDED_DIR / f"{recording_id}.json")
    utterances = data["utterances"]

    with psycopg.connect(DATABASE_URL, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM recordings WHERE id = %s", (recording_id,))

            cur.execute(
                """
                INSERT INTO recordings
                    (id, title, source_url, channel, license, language, duration_ms, audio_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    rec.id,
                    rec.title,
                    rec.source_url,
                    rec.channel,
                    rec.license,
                    rec.language,
                    round(rec.duration_sec * 1000),
                    rec.audio_path,
                ),
            )

            speaker_ids: dict[str, int] = {}
            speaker_names_seen = []
            for u in utterances:
                name = u["speaker"]
                if name not in speaker_names_seen:
                    speaker_names_seen.append(name)
            # diarization label isn't preserved post-chunk; recover via speaker_words cache
            from ingest.common import SPEAKER_WORDS_DIR

            sw = load_json(SPEAKER_WORDS_DIR / f"{recording_id}.json")
            name_to_label = {v: k for k, v in sw["speaker_map"].items()}

            for name in speaker_names_seen:
                label = name_to_label.get(name, name)
                cur.execute(
                    """
                    INSERT INTO speakers (recording_id, diarization_label, display_name)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (recording_id, label, name),
                )
                speaker_ids[name] = cur.fetchone()[0]

            for u in utterances:
                cur.execute(
                    """
                    INSERT INTO utterances
                        (recording_id, speaker_id, seq, start_ms, end_ms,
                         text_raw, text_norm, context_text, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        recording_id,
                        speaker_ids[u["speaker"]],
                        u["seq"],
                        u["start_ms"],
                        u["end_ms"],
                        u["text_raw"],
                        u["text_norm"],
                        u["context_text"],
                        u.get("embedding"),
                        u.get("embedding_model"),
                    ),
                )
                utterance_id = cur.fetchone()[0]

                for idx, w in enumerate(u["words"]):
                    cur.execute(
                        """
                        INSERT INTO words (utterance_id, idx, word, start_ms, end_ms, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (utterance_id, idx, w["word"], w["start_ms"], w["end_ms"], w.get("confidence")),
                    )

        conn.commit()

    log.info(
        f"[{recording_id}] loaded: {len(speaker_ids)} speakers, {len(utterances)} utterances -> Postgres"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording_id", nargs="?")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for rec in load_manifest():
            load_recording(rec.id)
    elif args.recording_id:
        load_recording(args.recording_id)
    else:
        ap.error("provide a recording_id or --all")


if __name__ == "__main__":
    main()
