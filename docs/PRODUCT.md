# NUVIX: product overview

Written for: anyone evaluating or using NUVIX, from product and engineering to first-time users.

## What NUVIX is

NUVIX is a search tool for recorded conversations. Give it a few words you remember, and it
returns the recording, the speaker and the exact moment they said it, then plays that moment.
It works on keywords, misspellings, paraphrases and across English and Hindi, and it runs
entirely on your own machine.

## The problem it solves

Finding a remembered moment in recorded audio means scrubbing through an hour of recording,
or listening to the whole thing again. Keyword search over a transcript misses misspellings,
paraphrases and speech recognition errors. Multilingual conversations, where people switch
between English and Hindi mid-sentence, break most single-language tools.

## Who it is for

- **Media teams** finding clips in podcasts and interviews.
- **Legal and compliance teams** finding what was said, by whom and when, without sending
  sensitive recordings to a cloud service.
- **Customer-experience teams** finding calls where a topic came up, however it was phrased.
- **Researchers** comparing what each participant said across many interviews.
- **Individuals** searching their own meeting notes and voice recordings.

## Alternatives and how NUVIX differs

The market already has tools close to parts of this. They are useful to know about, and they
show where NUVIX fits.

**Meeting and personal recorders**
- **Otter.ai, Fireflies.ai:** searchable, speaker-labelled meeting transcripts that jump to the
  timestamp. These are the nearest to NUVIX's search-and-play flow.
- **Apple Voice Memos, Google Recorder:** on-device transcripts you can search on your phone.

**Sales and call intelligence** (closest to the customer-experience use case)
- **Gong, Chorus (ZoomInfo), Clari:** search calls by topic or competitor mention, and filter by
  who spoke, such as rep versus customer.
- **Observe.AI, Balto, CallMiner:** contact-centre compliance and quality checks.

**Legal**
- **Everlaw, Relativity:** searchable transcripts linked to the source audio or video, used in
  discovery.

**Research**
- **Dovetail, Condens:** tag and search interview transcripts across a study, close to the
  research use case.

**Transcription services**
- **Rev, Sonix, Trint, Descript:** transcription with editing and search. Descript lets you edit
  audio by editing the transcript.

**Open source, closest to our stack**
- **WhisperX:** Whisper with word-level timestamps and speaker diarization. It is the same family
  of tools NUVIX uses, but it is a transcription pipeline, not a search product.

**Where NUVIX is different**
- **Local and private:** the audio and index never leave your machine. Most products above are
  cloud services, which matters for legal and healthcare use.
- **Hindi and English together:** most tools are strongest in English, so multilingual search is
  less common.
- **Measured retrieval:** NUVIX publishes recall, MRR and an ablation. Commercial tools rarely
  publish accuracy numbers.

The incumbents are ahead on polish, calendar and CRM integrations, speaker renaming and mobile
apps. NUVIX competes on privacy and multilingual search, not on features.

## What it does

- **Hybrid search:** exact keywords, fuzzy matching for misspellings, and semantic matching
  for meaning, combined and reranked.
- **Speaker and timestamp on every result**, with the matching text highlighted.
- **Play from the match:** the audio jumps to the moment and stops at the end of that line.
- **Method controls:** turn keyword, fuzzy, semantic and rerank on or off to see what each one
  contributes.
- **Evaluation built in:** recall, MRR and nDCG against a labelled query set, with an ablation
  across configurations.

## Current status

- Five recordings indexed (one English, four Hindi or Hinglish).
- Measured on 44 labelled queries: Recall@5 0.913, MRR 0.909, nDCG@10 0.886. These were tuned
  on the same query set, so treat them as optimistic. See the [README](../README.md) for the
  full results and limits.
- Live demo: https://mohmmadaslam--nuvix-web.modal.run (free CPU tier; the first request after
  a quiet period takes about 45 seconds).

## Not yet done

- Word error rate and speaker-attribution accuracy are not yet measured.
- One recording has unreliable speaker labels and is searchable but not evaluated.
- Access control, persistent storage and monitoring are needed before a public rollout.

## Where to go next

- [README](../README.md): setup, results and limitations.
- [ARCHITECTURE.md](ARCHITECTURE.md): design and rationale.
- [AGENT_LOG.md](AGENT_LOG.md): how the system was built, including the decisions and bugs along the way.
