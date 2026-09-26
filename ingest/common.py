"""Shared paths, manifest loading, and small utilities for the ingestion pipeline."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/audio_search")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

MANIFEST_PATH = ROOT / "data" / "manifest.yaml"

PIPELINE_DIR = ROOT / "data" / "pipeline"
TRANSCRIPTS_DIR = PIPELINE_DIR / "transcripts"
DIARIZATION_DIR = PIPELINE_DIR / "diarization"
SPEAKER_WORDS_DIR = PIPELINE_DIR / "speaker_words"
UTTERANCES_DIR = PIPELINE_DIR / "utterances"
EMBEDDED_DIR = PIPELINE_DIR / "embedded"

for d in (TRANSCRIPTS_DIR, DIARIZATION_DIR, SPEAKER_WORDS_DIR, UTTERANCES_DIR, EMBEDDED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "sentence_transformers", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


@dataclass
class Recording:
    id: str
    filename: str
    title: str
    language: str
    duration_sec: float
    num_speakers: int
    speakers: list[str]
    audio_path: str
    source_url: str | None = None
    channel: str | None = None
    license: str | None = None
    notes: str | None = None

    @property
    def audio_abspath(self) -> Path:
        return ROOT / self.audio_path


def load_manifest() -> list[Recording]:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    recordings = []
    for r in data["recordings"]:
        speakers = r.get("speakers", [])
        if isinstance(speakers, str):
            speakers = [speakers]
        recordings.append(
            Recording(
                id=r["id"],
                filename=r["filename"],
                title=r["title"],
                language=r["language"],
                duration_sec=float(r["duration_sec"]),
                num_speakers=int(r["num_speakers"]) if str(r["num_speakers"]).isdigit() else 2,
                speakers=list(speakers),
                audio_path=r["audio_path"],
                source_url=r.get("source_url"),
                channel=r.get("channel"),
                license=r.get("license"),
                notes=r.get("notes"),
            )
        )
    return recordings


def get_recording(recording_id: str) -> Recording:
    for r in load_manifest():
        if r.id == recording_id:
            return r
    raise KeyError(f"No recording with id={recording_id!r} in manifest")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
