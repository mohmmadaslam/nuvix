"""Deploy NUVIX to Modal (https://modal.com).

    python -m deploy.build_space                 # stage the bundle (DB dump, audio, code)
    modal deploy deploy/modal_app.py             # prints the public URL

One container runs Postgres 17 + pgvector, the search models on a T4 GPU and the demo
server. It scales to zero when idle (the first search after a pause waits for a cold
start) and is capped at one container so cost cannot run away.
"""
from pathlib import Path
import subprocess

import modal

BUNDLE = Path(__file__).resolve().parent / "build" / "space"

app = modal.App("nuvix")

image = (
    modal.Image.from_registry("pgvector/pgvector:pg17", add_python="3.11")
    .entrypoint([])  # drop the postgres image's docker-entrypoint
    .pip_install(
        "torch",
        "sentence-transformers==6.1.0",
        "transformers==5.17.0",
        "tokenizers==0.23.2",
        "huggingface_hub==1.33.0",
        "numpy==2.5.3",
        "psycopg[binary]==3.3.6",
        "pgvector==0.5.0",
        "python-dotenv==1.2.3",
        "PyYAML==6.0.3",
    )
    .env({
        "HF_HOME": "/models",
        "DATABASE_URL": "postgresql://postgres@127.0.0.1:5432/audio_search",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # Bake the models into the image so a cold start does not download ~4.5 GB.
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')\""
    )
    # Local files go last so editing the app does not rebuild the layers above.
    .add_local_dir(BUNDLE, remote_path="/app")
    .add_local_file(Path(__file__).resolve().parent / "modal_start.sh", remote_path="/app/modal_start.sh")
)


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=300,   # stay warm 5 min after the last request
    max_containers=1,       # one container: caps cost and keeps the DB copy consistent
    timeout=3600,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=600)
def web():
    subprocess.Popen(["bash", "/app/modal_start.sh"])
