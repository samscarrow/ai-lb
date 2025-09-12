#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

export COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1

function cleanup() {
  docker compose down -v || true
}
trap cleanup EXIT

MODEL_ID="m"

echo "[ci-smoke] Bringing up stack..."
ROUTING_STRATEGY=ROUND_ROBIN \
STICKY_SESSIONS_ENABLED=true \
SCAN_HOSTS='ai_lb_node1:9999,ai_lb_node2:9999' \
HEDGING_SMALL_MODELS_ONLY=false \
HEDGING_MAX_DELAY_MS=0 \
docker compose --profile full up -d --build

echo "[ci-smoke] Waiting for LB to respond (in-network)..."
for i in {1..90}; do
  if docker ps --format '{{.Names}}' | grep -q '^ai_lb_monitor$' && \
     docker exec -i ai_lb_monitor curl -sf http://ai_lb_load_balancer:8000/metrics >/dev/null; then
    break
  fi
  sleep 1
done

echo "[ci-smoke] Seeding Redis model/node state..."
docker exec -i ai_lb_redis redis-cli SADD nodes:healthy 'ai_lb_node1:9999' 'ai_lb_node2:9999' >/dev/null
docker exec -i ai_lb_redis redis-cli SET node:ai_lb_node1:9999:models '{"object":"list","data":[{"id":"'"$MODEL_ID"'","object":"model"}]}' >/dev/null
docker exec -i ai_lb_redis redis-cli SET node:ai_lb_node2:9999:models '{"object":"list","data":[{"id":"'"$MODEL_ID"'","object":"model"}]}' >/dev/null
docker exec -i ai_lb_redis redis-cli SET session:s:$MODEL_ID ai_lb_node1:9999 >/dev/null

echo "[ci-smoke] Forcing capacity on node1; freeing node2..."
docker exec -i ai_lb_redis redis-cli SET node:ai_lb_node1:9999:maxconn 1 >/dev/null
docker exec -i ai_lb_redis redis-cli SET node:ai_lb_node1:9999:inflight 1 >/dev/null
docker exec -i ai_lb_redis redis-cli DEL node:ai_lb_node2:9999:maxconn >/dev/null
docker exec -i ai_lb_redis redis-cli SET node:ai_lb_node2:9999:inflight 0 >/dev/null

round=1
ok=0
while (( round <= 3 )); do
  echo "[ci-smoke] Round $round: firing 10 non-stream requests to trigger hedging..."
  for i in $(seq 1 10); do
    docker exec -i ai_lb_monitor curl -s -o /dev/null -X POST http://ai_lb_load_balancer:8000/v1/chat/completions \
      -H 'content-type: application/json' \
      -H 'x-session-id: s' \
      --data '{"model":"'"$MODEL_ID"'","messages":[{"role":"user","content":"hedge attempt '"$i"' (round '"$round"')"}],"max_tokens":8,"stream":false}' || true
  done
  sleep 2
  echo "[ci-smoke] Reading metrics..."
  METRICS="$(docker exec -i ai_lb_monitor curl -s http://ai_lb_load_balancer:8000/metrics || true)"
  REQS=$(echo "$METRICS" | awk '/^ai_lb_requests_total /{print $2}')
  HEDGES=$(echo "$METRICS" | awk '/^ai_lb_hedges_total /{print $2}')
  WINS_MODEL=$(echo "$METRICS" | awk -v m="$MODEL_ID" '$1 ~ /^ai_lb_hedge_wins_total/ && $0 ~ "model=\""m"\"" {print $2}')

  echo "[ci-smoke] ai_lb_requests_total=${REQS:-NA}"
  echo "[ci-smoke] ai_lb_hedges_total=${HEDGES:-0}"
  echo "[ci-smoke] ai_lb_hedge_wins_total{model=\"$MODEL_ID\"}=${WINS_MODEL:-0}"

  if [[ -n "$HEDGES" ]] && (( ${HEDGES%%.*} >= 1 )) && [[ -n "${WINS_MODEL:-}" ]] && (( ${WINS_MODEL%%.*} >= 1 )); then
    ok=1
    break
  fi
  round=$((round+1))
done

if (( ok == 1 )); then
  echo "[ci-smoke] OK: Hedging counters present and > 0"
else
  echo "[ci-smoke] FAIL: Hedging counters not observed after retries" >&2
  exit 1
fi
