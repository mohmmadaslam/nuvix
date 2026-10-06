# NUVIX: product details

Written for: reviewers and team members who need the problem, scope and deliverables.

## Problem statement

**Theme:** Multimodal AI: Audio search
**Name:** Effective retrieval from audio transcripts

Audio recordings across conversations each contain two speakers. The question is how to
achieve effective hybrid search across them, combining keyword and semantic search. Which
diarization, database, embedding and indexing strategies give strong retrieval quality at
scale? And what metrics and evaluation matter before a system like this goes to production?

## What the task asks for

1. **Golden dataset:** 5 to 6 audio files of 8 to 10 minutes each, each with a unique pair of
   two speakers. Interviews or podcasts are allowed.
2. **Hybrid search strategy:** users must be able to search for exact words that were spoken,
   as well as terms that are semantically similar to their query.
3. **Local implementation:** generate transcripts from each file and index them. A search
   returns the containing file, timestamp and speaker. Embedding generation and indexing run
   locally; transcription may use a hosted or local service.
4. **Evaluation:** automated tests that measure recall@k against a small labelled query set.

## Tech stack

- Python or TypeScript (Python used)
- A relational database with embedding support, such as Postgres with pgvector (Postgres 17
  with pgvector used)

## Restrictions

- A coding agent may be used, but the team must explain how it was directed to reach the
  solution. See [AGENT_LOG.md](AGENT_LOG.md).

## Submission requirements

- A write-up in Markdown or PDF covering: the engineering design and rationale, success
  criteria, achievement against those criteria, and limitations. See the
  [README](../README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
- A disclosure of coding-agent use, with how the agent was directed. See
  [AGENT_LOG.md](AGENT_LOG.md).
- Source code, the golden dataset and the tests in a GitHub or GitLab repository.
