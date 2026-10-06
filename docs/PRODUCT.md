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
