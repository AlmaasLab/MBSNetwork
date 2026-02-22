#!/usr/bin/env bash
set -euo pipefail

set -a
source ../.env
set +a

cat db.dump.part-* | docker exec -i mbs-postgres \
  pg_restore -U "$PG_USER" -d "$PG_DB" --clean --if-exists
