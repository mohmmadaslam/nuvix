# NUVIX — Hybrid Intelligence for Audio

Hybrid search over two-speaker audio transcripts. Built for the G2 AI Hiring Hackathon —
Problem Statement 1 (Multimodal AI: Audio search).

**Status: in progress.** This README is being written alongside the remaining engineering
work. Sections marked `[TODO]` depend on the full 5-file corpus finishing ingestion and are
filled in once that's done — everything else reflects the actual current state of the
system, not a plan.

Full design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Coding-agent collaboration disclosure: [docs/AGENT_LOG.md](docs/AGENT_LOG.md).

## What this is

A locally-run hybrid (keyword + semantic) search system over transcribed, diarized audio
recordings. Given a text query, it returns the containing file, timestamp, and speaker for
matching moments — combining exact/fuzzy lexical matching with semantic embedding search,
fused and reranked, entirely on local infrastructure (Postgres + pgvector, open-weight
models, no external API calls at query time).

## Engineering design (summary — see docs/ARCHITECTURE.md for full detail and rationale)

```
Video -> ffmpeg (extract audio) -> faster-whisper large-v3 (transcribe, word timestamps)
       -> pyannote 3.1 (diarize, num_speakers=2) -> word-to-speaker assignment
       -> chunk into utterances (turn -> sentence-bounded chunks, degenerate-turn merging)
       -> BAAI/bge-m3 (embed, local, MPS-accelerated) -> Postgres (pgvector + pg_trgm)

Query -> lexical (Postgres FTS, 'simple' config) + fuzzy (trigram word_similarity)
       + semantic (HNSW cosine) -> Reciprocal Rank Fusion -> BAAI/bge-reranker-v2-m3
       -> highlighted, timestamped, speaker-attributed results
```

**Key design decisions and why** (each one changed from an initial default after review —
see docs/AGENT_LOG.md for the full back-and-forth):

- **`BAAI/bge-m3` embeddings, not an English-only model.** The dataset is deliberately mixed
  (1 English, 4 Hindi/Hinglish code-switched) specifically to test cross-lingual semantic
  retrieval, which an English-only embedding model couldn't support.
- **A single `simple` (non-stemming) Postgres full-text-search config**, not per-language
  stemming configs. Devanagari and Latin script coexist within the same transcripts
  (code-switching); one script-agnostic config avoids branching every query by language.
- **Native word-level Whisper timestamps**, not a separate forced-aligner. The original plan
  used a dedicated wav2vec2 aligner, but that's a second per-language model of uneven
  quality (solid for English, less so for Hindi) — dropped in favor of one model working
  uniformly across the whole mixed-language corpus.
- **Reciprocal Rank Fusion + cross-encoder reranking**, with the reranker's absolute score
  only trusted when no independent lexical/fuzzy method already found the same result via
  literal matching (see "Bugs found" below for why).

## Golden dataset

| Recording | Language | Speakers | Duration | Note |
|---|---|---|---|---|
| Elon Musk × Sam Altman (YC) | en | Sam Altman, Elon Musk | 19.5 min (full) | |
| Hema Malini podcast (Hindi Rush) | hi (code-switched) | Harshvardhan Pathak, Hema Malini | 17.9 min (full) | |
| Illuminati explanation (Ranveer Show) | hi (code-switched) | Ranveer Allahbadia, Ankit Awasthi | 6 min (trimmed from 14.3 min) | |
| Raj Shamani × Dr. Vikas Diviyakirti | hi (code-switched) | Raj Shamani, Dr. Vikas Diviyakirti | 6 min (trimmed from 10.4 min) | |
| Gaur Gopal Das × RJ Kartik | hi (code-switched) | RJ Kartik, Gaur Gopal Das | 6 min (trimmed from 15.75 min) | |

Full sourcing, licensing, and per-decision rationale: [data/manifest.yaml](data/manifest.yaml).

A 6th recording (a BBC World Service episode) was dropped from the original 6-file set — its
description named 5+ distinct voices (a produced multi-voice segment), which didn't satisfy
the task's two-speaker requirement. 5 files is within the task's stated 5-6 file range.

## Success criteria

Defined in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#7-evaluation) before any results were
measured:

| Metric | Target |
|---|---|
| Hybrid+rerank Recall@5 | ≥ 0.85 |
| MRR | ≥ 0.7 |
| WER (per language, reported separately) | < 15% |
| Speaker attribution accuracy | > 90% |
| p95 query latency (pre-rerank) | < 300ms |

## Results

### Retrieval quality (13-query gold set, `elon-musk-build-the-future` only — see limitations)

Measured via `eval/run_eval.py`, an ablation across 5 retrieval configurations:

| Config | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|
| lexical only | 0.385 | 0.385 | 0.385 |
| fuzzy only | 0.538 | 0.538 | 0.509 |
| semantic only | 0.769 | 0.769 | 0.750 |
| hybrid (RRF, no rerank) | 0.769 | 0.769 | 0.760 |
| **hybrid + rerank (default)** | **0.923** | **0.923** | **0.923** |

Hybrid+rerank beats every single method alone and clears the success-criteria target
(Recall@5 ≥ 0.85) on this file. Per-category breakdown (hybrid+rerank): negative queries
(no true answer) scored 1.00 recall — zero false positives; paraphrase queries (the hardest
category, testing pure semantic matching with no literal word overlap) scored 0.67, the
weakest category as expected.

`[TODO]` Full corpus results (all 5 files) once ingestion completes: this includes the
`cross_file_topic` and `cross_lingual_semantic` query categories, which can only be tested
with more than one file in the corpus.

`[TODO]` WER measurement against hand-corrected gold transcript slices — not yet performed;
see Limitations.

`[TODO]` Speaker attribution accuracy, measured (not just spot-checked) across the full
corpus.

### Query latency

`[TODO]` — p50/p95/p99 measurement once the full corpus is loaded and a representative query
mix can be run.

## Limitations

Documented as they were found, not smoothed over:

- **File lengths deviate from the task's 8-10 minute target in both directions.** 2 files
  kept at their full original length (17.9-19.5 min — a prior decision, made before
  processing-time constraints became clear); 3 files trimmed to a fixed 6 minutes (a later,
  time-constrained decision to keep CPU-only local transcription tractable on the dev
  hardware). See [data/manifest.yaml](data/manifest.yaml) for the full decision log.
- **CPU-only transcription is slow and doesn't reflect production hardware.** `large-v3` on
  a fanless Apple M4 laptop ran below realtime, with one 17.9-minute file taking over an
  hour. In production this would run on GPU infrastructure or a hosted ASR API (explicitly
  permitted by the task) — see docs/ARCHITECTURE.md Section 8.
- **Diarization struggles with fast turn-taking**, a known limitation of the underlying
  model, not this pipeline's chunking logic. Measured (not assumed): 5 of 3092 words (0.16%)
  in the Elon Musk file were misattributed across a rapid interjection boundary. A related,
  more severe case (a diarization flicker fragmenting one sentence into a 20ms orphan
  utterance) was caught via manual testing and fixed by merging degenerate short turns at
  the chunking stage.
- **Diarization failed badly on one file.** `hema-malini-podcast`: 94%+ of the 17.9-minute
  file was collapsed into a single speaker cluster (Hema Malini credited with only ~18
  seconds total, first appearing at 16:53), despite content clearly containing both question
  and first-person-answer turns throughout. Checked and ruled out a channel-separation cause
  (source video's L/R stereo correlation is 0.795 — a normal mixed-down track, not separate
  per-speaker mics); this is a genuine model failure on this file's audio, not a fixable
  pipeline bug. Accepted as a documented limitation rather than chased further: speaker
  labels for this recording are unreliable and it's excluded from
  speaker-attribution-dependent gold queries/metrics, while remaining usable for
  content/topic-only queries.
- **The cross-encoder reranker is not reliably calibrated across topics.** Found via manual
  testing: a genuine top-ranked match (agreed on by all 3 retrieval methods) scored 0.006 on
  an absolute 0-1 scale, while an equivalent-style query on different content scored 0.166.
  Mitigated by not letting the reranker's absolute score override a result any independent
  method already found via literal matching — but this is a mitigation, not a fix to the
  underlying model behavior.
- **The gold query set currently covers only 1 of 5 files** and lacks the cross-file/
  cross-lingual categories the architecture calls for — both blocked on the remaining 4
  files finishing ingestion, not a design gap.
- **Forcing `language='hi'` transliterates English words into Devanagari script**, rather
  than switching to Latin script for code-switched English content. Confirmed by inspecting
  the actual transcript: e.g. "documentary" is rendered phonetically as "डॉक्यूमेंटरी", not
  "documentary". This is a side effect of forced-language decoding, not a transcription
  error — but it means literal-English-spelling gold queries won't lexically match these
  passages, and likely inflates WER for the English-origin portions of code-switched speech.
  Affects the 4 Hindi/Hinglish files; noted here for whoever writes their gold query sets.
- **No WER measurement yet.** The architecture calls for hand-correcting ~2 minutes of gold
  transcript per file; this requires listening to the actual audio, which hasn't been done.
- **A deliberate scope decision, not a limitation of the retrieval design:** the query
  parser (phrase quoting, `speaker:`/`file:` filters) and adjacent-hit merging were deferred
  in favor of validating the core hybrid-retrieval mechanism first. `search/query.py` always
  runs all three retrieval legs rather than selectively skipping trigram search for
  non-rare terms, per the architecture's original optimization plan.
- **50 queries (the architecture's target gold-set size) is a small evaluation set** even
  once complete — results should be read with that in mind, not over-interpreted as
  precise population statistics.

## Repo layout

```
data/           manifest, videos, extracted audio, golden query labels
db/             Postgres schema
ingest/         transcription -> diarization -> chunking -> embedding -> load pipeline
search/         hybrid retrieval (lexical + fuzzy + semantic + RRF + rerank)
eval/           recall@k / MRR / nDCG scorer, ablation runner
tests/          pytest suite (unit tests for metrics, integration tests against the gold set)
docs/           architecture design doc, agent-collaboration disclosure
requirements.in / requirements.txt   pinned Python dependencies
```

## Running it

```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
brew install postgresql@17 pgvector && brew services start postgresql@17
createdb audio_search && psql -d audio_search -f db/schema.sql
cp .env.example .env   # fill in HF_TOKEN (see .env.example for the gated-model URLs to accept)

python -m ingest.run_pipeline <recording_id>     # per file, or --all
python -m search.query "your query here" --k 5
python -m eval.run_eval                          # ablation report against the gold set
python -m pytest tests/                          # automated recall@k / MRR tests
```
