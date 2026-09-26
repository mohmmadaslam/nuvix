#!/usr/bin/env bash
# Container start: build a fresh Postgres cluster, load the database dump, run the server.
# The database is rebuilt on every start (it is 13 MB), so the container holds no state.
set -euo pipefail

export PGDATA=/home/user/pgdata
rm -rf "$PGDATA"
initdb -D "$PGDATA" -U user --auth=trust -E UTF8 --locale=C.UTF-8 >/dev/null
{
  echo "listen_addresses = '127.0.0.1'"
  echo "unix_socket_directories = '/tmp'"
  echo "shared_buffers = 128MB"
} >> "$PGDATA/postgresql.conf"
pg_ctl -D "$PGDATA" -l /tmp/postgres.log -w start >/dev/null

createdb -h 127.0.0.1 audio_search
psql -h 127.0.0.1 -d audio_search -v ON_ERROR_STOP=1 -q -f db/nuvix.sql >/dev/null
echo "database loaded: $(psql -h 127.0.0.1 -d audio_search -Atc 'select count(*) from utterances') utterances"

exec python -m api.server --host 0.0.0.0 --port 7860
