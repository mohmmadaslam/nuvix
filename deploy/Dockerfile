# NUVIX on a Hugging Face Docker Space: Postgres 17 + pgvector, the search models
# and the demo server in one container. CPU only (free Spaces have no GPU).
FROM pgvector/pgvector:pg17

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Spaces as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/usr/lib/postgresql/17/bin:/home/user/venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    DATABASE_URL=postgresql://localhost:5432/audio_search \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app

RUN python3 -m venv /home/user/venv
COPY --chown=user requirements-runtime.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-runtime.txt

# Bake the models into the image so a cold start does not download ~4.5 GB.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY --chown=user . .

EXPOSE 7860
ENTRYPOINT []
CMD ["bash", "start.sh"]
