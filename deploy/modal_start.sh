#!/usr/bin/env bash
# Start script for the Modal container. Modal runs as root and Postgres refuses to,
# so the cluster is created and run as the image's `postgres` user. The database is
# rebuilt from the dump on every container start (13 MB), so the container holds no state.
set -euo pipefail

PGBIN=/usr/lib/postgresql/17/bin
export PGDATA=/var/lib/nuvix-pg
rm -rf "$PGDATA"; mkdir -p "$PGDATA"; chown postgres:postgres "$PGDATA"

runuser -u postgres -- "$PGBIN/initdb" -D "$PGDATA" -A trust -E UTF8 --locale=C.UTF-8 >/dev/null
{
  echo "listen_addresses = '127.0.0.1'"
  echo "shared_buffers = 128MB"
} >> "$PGDATA/postgresql.conf"
runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" -l /tmp/postgres.log -w start >/dev/null

"$PGBIN/createdb" -h 127.0.0.1 -U postgres audio_search
"$PGBIN/psql" -h 127.0.0.1 -U postgres -d audio_search -v ON_ERROR_STOP=1 -q -f /app/db/nuvix.sql >/dev/null
echo "database loaded: $("$PGBIN/psql" -h 127.0.0.1 -U postgres -d audio_search -Atc 'select count(*) from utterances') utterances"

# CPU reranking is slow; scoring 15 candidates instead of 30 measured Recall@5 0.905 vs 0.913
# (MRR 0.920 vs 0.909) on the 44-query gold set, for about half the time.
export NUVIX_RERANK_POOL=15

cd /app
exec python -m api.server --host 0.0.0.0 --port 8000
