---
title: NUVIX Audio Search
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# NUVIX - Hybrid Intelligence for Audio

Live demo of a local hybrid (keyword + fuzzy + semantic) search system over five
two-speaker audio recordings, built as a solution to Problem Statement 1.
Search returns the file, timestamp and speaker, and lets you play the audio at the match.

This Space runs on CPU, so searches with the cross-encoder rerank on take several seconds;
turn Rerank off in the UI for instant results.

Source code, design and evaluation: https://github.com/mohmmadaslam/nuvix
