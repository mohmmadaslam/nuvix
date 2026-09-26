-- Schema for hybrid audio-transcript search.
-- Postgres 17 + pgvector (0.8.6) + pg_trgm.
-- Mirrors docs/ARCHITECTURE.md Section 4.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- One row per source recording (one per manifest.yaml entry).
CREATE TABLE recordings (
    id              TEXT PRIMARY KEY,           -- matches manifest id, e.g. 'elon-musk-build-the-future'
    title           TEXT NOT NULL,
    source_url      TEXT,
    channel         TEXT,
    license         TEXT,
    language        TEXT NOT NULL,              -- 'en' | 'hi' (may be code-switched)
    duration_ms     INTEGER NOT NULL,
    video_path      TEXT,
    audio_path      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per distinct diarized speaker within a recording.
CREATE TABLE speakers (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    recording_id        TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    diarization_label   TEXT NOT NULL,          -- e.g. 'SPEAKER_00'
    display_name        TEXT NOT NULL,          -- e.g. 'Sam Altman' (from manifest first-appearance order)
    UNIQUE (recording_id, diarization_label)
);

-- One row per retrieval-unit utterance (the thing search actually returns).
CREATE TABLE utterances (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    recording_id    TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    speaker_id      BIGINT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,           -- order within the recording, 0-based
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,

    text_raw        TEXT NOT NULL,              -- as transcribed, for display
    text_norm       TEXT NOT NULL,              -- disfluencies/numbers normalized, for indexing
    context_text    TEXT NOT NULL,              -- text_norm + neighboring utterances, embedding input only

    tsv_simple      TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text_norm)) STORED,

    embedding       VECTOR(1024),               -- bge-m3 dense output
    embedding_model TEXT,                       -- e.g. 'BAAI/bge-m3' - versioned for future re-embedding

    UNIQUE (recording_id, seq),
    CHECK (end_ms > start_ms)
);

-- One row per ASR word, for exact-timestamp highlighting of lexical hits.
CREATE TABLE words (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    utterance_id    BIGINT NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    idx             INTEGER NOT NULL,           -- word position within the utterance, 0-based
    word            TEXT NOT NULL,
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    confidence      REAL,

    UNIQUE (utterance_id, idx)
);

-- --- Indexes ---------------------------------------------------------------

-- Semantic search
CREATE INDEX utterances_embedding_hnsw
    ON utterances USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Lexical search (language-agnostic 'simple' config - see architecture doc
-- Section 4 for why: single index works across English and Hindi/Devanagari
-- without per-language branching)
CREATE INDEX utterances_tsv_simple_gin
    ON utterances USING gin (tsv_simple);

-- Fuzzy matching (ASR-damaged proper nouns; script-agnostic)
CREATE INDEX utterances_text_norm_trgm
    ON utterances USING gin (text_norm gin_trgm_ops);

-- Timestamp-ordered scans for merging adjacent hits / fetching context
CREATE INDEX utterances_recording_start_idx
    ON utterances (recording_id, start_ms);

-- words lookup by utterance (already covered by FK + UNIQUE, explicit for clarity)
CREATE INDEX words_utterance_idx
    ON words (utterance_id, start_ms);
