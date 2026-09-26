# Architecture: hybrid search over two-speaker audio transcripts

Problem statement 1 from the G2 AI Hiring Hackathon: effective hybrid (keyword + semantic)
retrieval across two-speaker audio recordings, at file/timestamp/speaker granularity, evaluated
with recall@k against a labeled query set.

This document is the design spec: what we're building and why. It is also the spec handed to
the coding agent during implementation (see `docs/AGENT_LOG.md`, added once implementation
starts, for how it was directed).

## 1. System overview

```
                         +---------------- INGESTION (offline, batch) ----------------+
 videos/*.mp4 --> Extract  --> ASR + word   --> Diarize   --> Assign words --> Segment  --> Embed --> Postgres
 + manifest       audio        timestamps       (pyannote,    to speakers      & chunk      (bge-m3   (pgvector
   .yaml          (ffmpeg,     (faster-whisper  n=2)          + name map       (turns ->    local)    + FTS
                   16k mono)   large-v3,                                       utterances)            + trgm)
                               native word
                               timestamps)
                         +--------------------------------------------------------------+

                         +---------------- QUERY (online) --------------------------------+
 "query" --> Parse (phrases, --+--> Lexical (FTS/simple)   top-50 --+
             speaker:/file:)   +--> Fuzzy (pg_trgm)        top-20 --+--> RRF fuse --> Rerank --> Merge adjacent --> Results
                               +--> Semantic (HNSW cosine) top-50 --+   (k=60)      (cross-    + highlight        (file, speaker,
                                                                                     encoder)                      mm:ss, snippet)
                         +-----------------------------------------------------------------+

                         +---------------- EVALUATION -------------------------------------+
  golden queries (time-interval labels) --> run all retrieval modes --> Recall@k, MRR, nDCG
  gold transcript slices                 --> WER, speaker attribution accuracy, timestamp error
                         +-----------------------------------------------------------------+
```

## 2. Golden dataset

**6 recordings, sourced as full YouTube podcast/interview videos, each with two speakers:**

| id | language | duration | notes |
|---|---|---|---|
| bbc-social-media-bans | en | 19.8 min | BBC World Service podcast |
| elon-musk-build-the-future | en | 19.5 min | YC-style interview |
| hema-malini-podcast | hi (code-switched) | 17.9 min | Bollywood interview podcast |
| illuminati-hindi-explanation | hi (code-switched) | 14.3 min | confirmed two-person conversation |
| raj-shamani-vikas-diviyakirti | hi (code-switched) | 10.4 min | interview podcast |
| gaur-gopal-das-rj-kartik | hi (code-switched) | 15.75 min | interview/story podcast |

Recorded per-file in [`data/manifest.yaml`](../data/manifest.yaml): id, filename, title, source
URL, license, language, duration, speaker count, speaker names (filled in after diarization +
manual verification), notes.

**Deliberate mix, not incidental:** 2 English files + 4 Hindi/Hinglish (code-switched) files.
This is a stronger test of the hybrid design than an all-English set, because it forces:
- A multilingual embedding model rather than an English-only one.
- A script/language-agnostic lexical index rather than per-language stemming config.
- A genuinely interesting eval category: an English-language query retrieving a relevant
  segment from a Hindi file based on meaning alone (lexical search cannot do this at all).

**Documented deviations from the task spec, decided explicitly rather than discovered late:**
- *Length:* the spec asks for 8-10 minute files. These are the full original episodes
  (10.4-19.8 min), used at full length rather than trimmed. Decision made 2026-09-26 in favor
  of preserving complete, unedited conversations over spec-exact trimming.
- *Speaker count check:* one file's title ("ILLUMINATI: Detailed Hindi Explanation") reads like
  solo narration; confirmed by the person who selected it to be a genuine two-person
  conversation before inclusion.
- *Licensing:* these are downloaded YouTube episodes, not confirmed CC-licensed clips. Source
  URL and license fields in the manifest are TBD and must be resolved before any public
  GitHub/GitLab push of the raw audio/video files — committing full copyrighted media to a
  public repo is a separate risk from using it locally for development and evaluation.

**Gold transcript slices:** hand-correct ~2 minutes per file (12 minutes total) for WER and
speaker-attribution measurement. For files where an existing transcript/caption track is
available, correct that instead of transcribing from scratch — cheaper, but never treated as
ground truth as-is since auto-captions carry their own error rate and aren't diarized.

## 3. Ingestion pipeline

| Stage | Choice | Rationale |
|---|---|---|
| Audio extraction | ffmpeg: `*.mp4` -> 16 kHz mono WAV, loudness-normalized | Consistent input for ASR and diarization; strips video we don't need |
| ASR + word timestamps | **faster-whisper (large-v3)**, native word-level timestamps | One model across every language in the set (English and Hindi/code-switched); avoids depending on a separate forced-aligner whose per-language model quality varies (see below) |
| Diarization | **pyannote/speaker-diarization-3.1**, `num_speakers=2` | Audio-only, language-agnostic; telling it there are exactly 2 speakers materially reduces diarization errors |
| Word -> speaker | Assign each word to the speaker whose diarized segment overlaps it most; smooth single-word flips at turn boundaries | Prevents boundary words being misattributed |
| Speaker naming | Map SPEAKER_00/01 to real names via manifest's first-appearance order; manually overridable | Results show "Jane Doe", not a diarization label |

**Why native Whisper word timestamps instead of a separate forced-aligner (WhisperX +
wav2vec2):** the original design used a dedicated forced-aligner for tighter word timestamps.
That aligner is a second model chosen per language from a smaller pool of community CTC models
- solid for English, less consistent for Hindi. Given this dataset is 4/6 Hindi/Hinglish, a
second per-language dependency of uneven quality is the wrong trade. faster-whisper's built-in
word timestamps are marginally less precise than a dedicated aligner but come from one model,
uniformly, across every file in the set - simpler and more robust for a mixed-language corpus.

**Chunking** (the key retrieval decision):
1. **Turn:** everything one speaker says before the other speaks.
2. **Utterance (the retrieval unit):** long turns split at sentence boundaries into ~10-40s
   pieces (~150 tokens max); short turns stay whole.
3. **Context window (embedding-only):** the utterance plus its previous/next neighbor, since a
   short reply like "yeah, exactly" carries little meaning alone. Results still display the
   utterance's own timestamp and speaker, not the padded window.

Each utterance stores `text_raw` (display) and `text_norm` (disfluencies removed, numbers
normalized; indexing).

## 4. Storage: Postgres 16 + pgvector + pg_trgm

```
recordings(id, title, source_url, license, language, duration_ms, video_path, audio_path)
speakers(id, recording_id, diarization_label, display_name)
utterances(id, recording_id, speaker_id, seq, start_ms, end_ms,
           text_raw, text_norm, context_text,
           tsv_simple tsvector GENERATED,
           embedding vector(1024), embedding_model text)
words(utterance_id, idx, word, start_ms, end_ms, confidence)
```

Indexes:
- **HNSW** on `embedding` (`vector_cosine_ops`, m=16, ef_construction=64).
- **GIN** on `tsv_simple` — a single `simple` (non-stemming) tsvector config used uniformly
  across every file, English and Hindi/Devanagari alike. This trades away English stemming's
  small recall boost in exchange for not branching FTS config per language and not needing a
  language-detection step in the write path — the right trade for a genuinely mixed-language
  corpus with no reliable per-utterance language tag.
- **GIN trigram** on `text_norm` — catches ASR spelling mangling (proper nouns especially) and
  works the same regardless of script.
- **btree** on `(recording_id, start_ms)` for merging neighboring hits and fetching context.

`embedding_model` is stored per row so the corpus can be re-embedded with a new model later
without downtime.

## 5. Embeddings (local, multilingual)

- **`BAAI/bge-m3`** — multilingual (100+ languages, including Hindi), runs locally via
  sentence-transformers, and natively produces both dense and sparse representations from one
  model. Chosen over an English-only model (e.g. `bge-base-en-v1.5`) specifically because 4/6
  files are Hindi/Hinglish; one embedding model is used for every file rather than mixing
  models per language.
- **Reranker:** `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder, local), applied to the
  top ~30 fused results.

## 6. Query pipeline

1. **Parse the query:** `"quoted phrase"` -> exact phrase match; `speaker:` / `file:` filters;
   otherwise runs in hybrid mode.
2. **Retrieve in parallel** (one SQL statement, CTEs):
   - Lexical, top 50: `websearch_to_tsquery` against `tsv_simple`.
   - Fuzzy, top 20: trigram similarity, triggered for rare/capitalized terms.
   - Semantic, top 50: HNSW cosine, `ef_search=100`.
3. **Fuse with Reciprocal Rank Fusion** (k=60). Quoted-phrase or rare-exact-term queries weight
   the lexical ranking more heavily.
4. **Rerank** the top 30 with the cross-encoder, then cut to top k.
5. **Post-process:** merge same-file hits <5s apart; highlight exact word timestamps for
   lexical hits (via the `words` table), best-matching sentence for semantic hits.
6. **Result fields:** file, speaker name, `mm:ss-mm:ss`, highlighted snippet, score breakdown
   (lexical rank / vector rank / rerank score).

Interfaces: CLI (`search "..." --k 5`) and a FastAPI endpoint.

## 7. Evaluation

**~50 labeled queries**, spanning: exact keyword, exact phrase, paraphrase (no shared words),
ASR-damaged proper noun, speaker-filtered, cross-file topic, **cross-lingual semantic**
(English query retrieving a Hindi-file segment on meaning), and negative (no matching content).

- **Labels are time intervals**, not chunk IDs: `(recording_id, start_ms, end_ms, speaker)`. A
  hit counts as relevant if it overlaps a gold interval (±3s tolerance) — this keeps labels
  valid even if chunking, embedding model, or FTS config changes later.
- **Automated tests** (pytest + docker-compose Postgres):
  - Recall@1/5/10, MRR, nDCG@10 — overall and per query category.
  - Ablation: lexical-only vs. vector-only vs. hybrid-RRF vs. hybrid+rerank; asserts hybrid
    beats each single method and clears a Recall@5 threshold.
  - Upstream quality: WER on gold slices, speaker-attribution accuracy, timestamp error (ms).

| Metric | Target |
|---|---|
| Hybrid Recall@5 | >= 0.85 |
| MRR | >= 0.7 |
| WER (per language) | < 15% (Hindi/code-switched may run higher than English; reported separately) |
| Speaker attribution | > 90% |
| p95 query latency | < 300ms (pre-rerank) |

## 8. Production metrics (for the write-up)

- **Offline quality:** recall@k / nDCG per query category and per language; HNSW vs. exact
  recall.
- **Upstream quality:** WER, diarization error rate, speaker confusion — reported per language
  given the English/Hindi split.
- **Online:** zero-result rate, click position, reformulation rate, time-to-play, post-jump
  listen duration (>10s as a relevance proxy).
- **System:** query latency p50/p95/p99, ingestion real-time factor, index size per audio hour,
  embedding-model version field for future re-embedding.
- **Scaling:** queue-based ingestion workers (GPU for ASR/diarization), partitioning
  `utterances`, `halfvec` storage, cross-recording speaker identity via ECAPA embeddings,
  dedicated BM25 engine (pg_search/OpenSearch) if corpus grows past what `tsvector`+GIN handles
  well.

## 9. Repo layout

```
data/videos/, data/audio/, data/manifest.yaml, data/gold/{queries.yaml, transcripts/}
ingest/   (extract_audio, transcribe, diarize, assign_speakers, chunk, embed, load)
search/   (query parser, SQL retrieval, fusion, rerank, highlight)
api/, cli/
eval/     (metrics, ablation runner, report generator)
tests/    (pytest: recall@k thresholds, WER, speaker accuracy)
docker-compose.yml (postgres + pgvector)
docs/ARCHITECTURE.md (this file), docs/AGENT_LOG.md (agent-collaboration disclosure)
README.md (results, success criteria, limitations)
```

## Known risks and limitations

- **pyannote requires gated Hugging Face model access** — must be accepted before ingestion.
- **Overlapping/crosstalk speech** will be credited to one speaker only by diarization.
- **Code-switched Hindi/English ASR** is inherently harder than monolingual transcription;
  WER is expected to run higher on the 4 Hindi files than the 2 English files, reported
  separately rather than as one blended number.
- **File lengths exceed the spec's 8-10 min target** (10.4-19.8 min) — documented deviation,
  decided in favor of unedited full episodes (see Section 2).
- **Licensing of source videos is unresolved** — must be settled before any public repo push
  of raw media.
- **50 queries is a small eval set** — results reported with this caveat, not over-interpreted.
