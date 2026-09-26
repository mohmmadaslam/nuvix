# Coding agent collaboration disclosure

This project was built in collaboration with **Claude** (Anthropic's coding agent, running
as Claude Code), per the hackathon's disclosure requirement. This document summarizes how
the agent was directed and the substantive decisions made along the way — both by the
project owner and by the agent — rather than reproducing the full conversation transcript.

The pattern throughout: the agent proposed designs, wrote code, ran commands, and diagnosed
failures; the project owner made the calls that required human judgment — which problem to
solve, what data to use, licensing/privacy tradeoffs, and whether a proposed fix or workaround
was acceptable. Several of the most important fixes in this project came directly from the
project owner manually testing the running system and reporting results back to the agent,
not from the agent's own automated checks.

## 1. Problem selection and architecture design

The agent was given the hackathon's problem-statement PDF and asked to design a solution
architecture before writing any code (explicit instruction: "no code required at first
step"). The agent read all 3 problem statements and, after being asked to choose one,
selected **Problem Statement 1** (hybrid search over two-speaker audio transcripts), then
produced a full architecture document covering: golden dataset construction, ingestion
pipeline design (ASR, diarization, chunking, embeddings), storage schema, hybrid query
pipeline (lexical + semantic fusion + reranking), and an evaluation plan (recall@k, MRR,
nDCG, ablation study).

**Project owner's role:** approved the problem-statement choice, then asked follow-up
questions that changed the design before any implementation started:
- *"why can't we have some good model which can handle hindi video too?"* — pushed back on
  an initial English-only embedding model recommendation. The agent corrected course,
  switching to `BAAI/bge-m3` (multilingual) and re-architecting the lexical index around a
  script-agnostic `simple` Postgres FTS config instead of per-language stemming.
- Clarifying questions about whether the golden-dataset audio needed existing transcripts
  and whether every clip had to be in English — answered directly by the agent with
  reasoning, no design change needed.

## 2. Golden dataset curation

The project owner supplied 6 downloaded video files and, later, their original YouTube URLs.
The agent:
- Moved files into `data/videos/`, extracted audio via `ffmpeg`.
- Used `ffprobe` to check each file's duration and found none matched the task's 8-10 minute
  target (all were 10.4-19.8 min) — flagged this rather than silently proceeding.
- Matched the 6 YouTube URLs to files via oEmbed metadata, and cross-checked license status
  via the Hugging Face-style approach of querying actual metadata rather than assuming —
  confirming all 6 sources were standard-license YouTube uploads, not Creative Commons.
- Flagged that one file's description named 5+ distinct voices (a produced multi-voice BBC
  segment), which the project owner confirmed should be **dropped** rather than kept or
  swapped, leaving a 5-file dataset (within the task's 5-6 file range).
- Resolved speaker names per file via web search and video descriptions; one speaker's name
  (Hindi Rush's host) had no public source, so the project owner supplied it directly.

**Key project-owner decisions the agent could not make unilaterally:**
- **Licensing/repo visibility**: given none of the 6 sources were CC-licensed, the agent
  presented two options (public repo with derived-artifacts-only, or private repo with raw
  media committed) and the project owner chose the private-repo route, explicitly accepting
  that this repo must never be made public.
- **Git LFS vs. plain git** for the ~335MB of video files — project owner chose plain git.
- **Keep full-length files vs. trim to spec** — project owner initially chose to keep full
  original episode lengths rather than trim to 8-10 minutes, a documented deviation from the
  task spec.
- **Multilingual mix**: the final dataset (1 English, 4 Hindi/Hinglish code-switched) was a
  deliberate choice enabling a cross-lingual semantic-search test case, not an accident of
  which files happened to be available.

## 3. Environment setup

The agent's original plan called for Docker Compose (Postgres + pgvector). In practice:
Docker wasn't installed on the dev machine. The agent presented three options (native
Homebrew Postgres, Docker Desktop, Colima) and the project owner chose native Postgres for
speed of setup. The agent then discovered Homebrew's `pgvector` bottle only supports
Postgres 17/18, not the originally planned 16 — switched versions and documented why, since
this had no functional impact on the schema.

The agent could not accept pyannote's gated Hugging Face model terms on the project owner's
behalf — this required the project owner to create a Hugging Face account, generate a token,
and click through the model-access agreements themselves. The agent verified the token and
gated-model access programmatically afterward rather than taking it on faith.

## 4. Database schema

The agent proposed a concrete SQL schema (4 tables, 5 custom indexes) matching the
architecture document, and — at the project owner's request — presented it for review
*before* applying it to the database. Once approved, the agent applied it and verified the
result (table/index counts) before committing.

## 5. Ingestion pipeline: built, then debugged against real data

The agent wrote the full ingestion pipeline (`ingest/`: extract audio, transcribe via
faster-whisper, diarize via pyannote, assign speakers to words, chunk into utterances, embed
via bge-m3, load into Postgres) and, per the agreed plan, ran it end-to-end on **one file
first** (the English Elon Musk interview) before touching the other four, specifically so
problems could be caught and fixed cheaply.

**Bugs found and fixed during this first run**, all discovered by the agent reading error
output and diagnosing root causes (not by the project owner):
1. `pyannote.audio` 4.0.7's API had changed since the architecture was designed
   (`use_auth_token` → `token` kwarg; a new `DiarizeOutput` wrapper class replacing the old
   direct `Annotation` return type) — required two separate fixes across two failed runs.
2. pyannote 4.x's diarization pipeline now pulls a component from a *second* gated
   Hugging Face repo (`pyannote/speaker-diarization-community-1`) not mentioned in pyannote's
   original documentation the architecture was based on — required the project owner to
   accept a second model's terms.
3. Whisper hallucinated 31 zero-duration, low-confidence words (a known ASR failure mode),
   which violated the database's `CHECK (end_ms > start_ms)` constraint — the agent filtered
   these at the chunking stage rather than the transcription stage, to avoid needing to
   redo the expensive (~17 minute) transcription step.
4. `psycopg` required pgvector's `Vector()` wrapper type for vector-typed query parameters,
   not a plain Python list.

## 6. Search implementation: built, then debugged via the project owner's manual testing

The agent built a minimal hybrid search (lexical + fuzzy + semantic, RRF fusion) and
validated it against the one ingested file with a handful of test queries, demonstrating the
core value proposition (a semantic query with zero literal word overlap correctly retrieving
the right content; a real ASR-tokenization mismatch being rescued by the semantic leg where
lexical/fuzzy alone failed).

The project owner then asked to **round out the implementation with reranking and
highlighting**, and separately asked to **test it manually themselves**, requesting a list of
example queries to try. This manual-testing phase produced the project's most valuable bug
reports — each one below was found by the project owner running a query and reporting the
output back to the agent, who then diagnosed and fixed the root cause:

- **A 20-millisecond, one-word utterance ("We're") ranked #1** above genuinely relevant
  results for the query "going to space." The agent traced this to a diarization
  turn-boundary flicker that fragmented a real sentence across two speakers, and fixed it by
  adding a merge pass for degenerate short turns in the chunking stage — not by tuning the
  reranker, since the reranker was reacting sensibly to genuinely broken input.
- **"Stanford" (spelled correctly) returned zero results**, despite all three retrieval
  methods agreeing it was the top match. The agent diagnosed this as reranker miscalibration
  (the cross-encoder scored the correct passage at 0.006-0.007, while an equivalent-style
  query on different content scored 0.166) and fixed the *filtering logic* — not by lowering
  the threshold globally (which would have reopened the earlier noise problem), but by
  exempting lexical/fuzzy-corroborated hits from the reranker's absolute-score cutoff, since
  an independent method already found them via literal matching.
- **A genuinely correct result displayed as just the word "Today"** in the CLI, which looked
  like a ranking bug but was actually a display bug: Postgres's `ts_headline()` truncates to
  a handful of words when the full AND-query doesn't literally match, even for reasonably
  relevant fuzzy/semantic hits. Fixed by switching to `HighlightAll=true`, which shows full
  text unconditionally.

None of these three bugs were things the agent's own automated testing had caught — they
surfaced specifically because the project owner ran real queries against real data and
reported results that didn't look right, which the agent's own targeted smoke tests hadn't
happened to exercise.

## 7. Gold query set and evaluation framework

The agent drafted a 13-query gold set for the one ingested file, reading real timestamps
directly from Postgres rather than guessing, and explicitly asked the project owner to
verify relevance against the actual video (the agent has no ability to listen to audio
itself). The project owner requested minute:second conversions to make that verification
practical, then confirmed the set as validated.

The agent then built the recall@k/MRR/nDCG scorer and, before writing any pytest thresholds,
ran it against real data first — catching a genuine bug in its own initial nDCG
implementation (scores exceeding the mathematically-impossible value of 1.0, caused by
crediting the same gold interval multiple times) and fixing it before it could silently
under-test the system. Pytest thresholds were then set from the corrected baseline numbers,
with margin, rather than arbitrary guesses.

## 8. Processing-time constraints and dataset trimming

CPU-only `large-v3` transcription on the dev machine (a fanless Apple M4 MacBook Air) proved
far slower than expected — one Hindi file took 45+ minutes with no reliable way to estimate
completion. The agent explained *why* (no GPU acceleration for the CTranslate2 backend,
thermal throttling likely on fanless hardware) and explicitly recommended against
parallelizing multiple files on this single machine, since the CPU was already
near-saturated by one job and adding a second was more likely to slow both down via thermal
throttling than speed anything up. The project owner agreed to stay sequential.

When processing time became a real constraint, the project owner asked to trim the
remaining unprocessed files to 6 minutes each. The agent asked two clarifying questions
before acting (which files were in scope — excluding the one already 45+ minutes into
processing, to avoid discarding sunk compute — and which segment to extract), then
implemented the trim, updated the dataset manifest with a clear decision log, and confirmed
the original full-length audio remained untouched on disk for either file.

## Summary

The agent handled: architecture design, all code (ingestion pipeline, search, evaluation),
command execution, debugging, and root-cause diagnosis of every bug encountered. The project
owner handled: problem selection, dataset sourcing and curation calls, licensing/privacy
decisions, infrastructure choices where tradeoffs required human judgment, gold-label
verification against ground truth the agent cannot access (audio, video descriptions), and —
critically — the manual testing that surfaced the project's most significant retrieval-quality
bugs before they could have gone unnoticed into a final evaluation.
