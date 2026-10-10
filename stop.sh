#!/usr/bin/env bash
# Stop anything listening on the Outhit service ports.
for port in 8001 8201 8202 8203 8204 8205 8206; do
  pids=$(lsof -ti tcp:$port -sTCP:LISTEN || true)
  [ -n "$pids" ] && kill $pids && echo "stopped :$port"
done
