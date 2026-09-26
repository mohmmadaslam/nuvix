"""Assemble (and optionally upload) the Hugging Face Space bundle.

    python -m deploy.build_space                     # stage into deploy/build/space
    HF_TOKEN=hf_... python -m deploy.build_space --upload --repo <user>/nuvix

The bundle is: Dockerfile, Space README, runtime requirements, start.sh, the app code
(api/, search/, ingest/common.py), a pg_dump of the loaded database, the five audio files
and the saved evaluation report. The token is read from the environment only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "deploy" / "build" / "space"


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    d = ROOT / "deploy"
    shutil.copy(d / "Dockerfile", STAGE / "Dockerfile")
    shutil.copy(d / "SPACE_README.md", STAGE / "README.md")
    shutil.copy(d / "requirements-runtime.txt", STAGE / "requirements-runtime.txt")
    shutil.copy(d / "start.sh", STAGE / "start.sh")

    for pkg in ("api", "search"):
        shutil.copytree(ROOT / pkg, STAGE / pkg, ignore=shutil.ignore_patterns("__pycache__"))
    (STAGE / "ingest").mkdir()
    shutil.copy(ROOT / "ingest" / "__init__.py", STAGE / "ingest" / "__init__.py")
    shutil.copy(ROOT / "ingest" / "common.py", STAGE / "ingest" / "common.py")
    (STAGE / "eval").mkdir()
    shutil.copy(ROOT / "eval" / "report.json", STAGE / "eval" / "report.json")
    (STAGE / "data").mkdir()
    shutil.copytree(ROOT / "data" / "audio", STAGE / "data" / "audio")

    (STAGE / "db").mkdir()
    with open(STAGE / "db" / "nuvix.sql", "wb") as f:
        subprocess.run(
            ["pg_dump", "--no-owner", "--no-privileges", "--dbname", "audio_search"],
            check=True, stdout=f,
        )
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {STAGE} ({size / 1e6:.0f} MB)")


def upload(repo_id: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (a Hugging Face token with write access) in the environment")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy NUVIX demo")
    print(f"uploaded: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo", help="<hf-username>/<space-name>")
    args = ap.parse_args()
    stage()
    if args.upload:
        if not args.repo:
            raise SystemExit("--repo is required with --upload")
        upload(args.repo)
