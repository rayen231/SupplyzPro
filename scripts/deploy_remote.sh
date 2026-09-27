#!/usr/bin/env bash
# Runs ON the Brev instance. Unpacks the bundle and starts dashboard (8501), API (8000) and MLflow UI (5000).
# Usage: bash deploy_remote.sh ~/supplyzpro.tar.gz
set -euo pipefail
BUNDLE="${1:-$HOME/supplyzpro.tar.gz}"
APP="$HOME/supplyzpro"
mkdir -p "$APP" && tar -xzf "$BUNDLE" -C "$APP"
[ -f "$HOME/.supplyzpro.env" ] && cp "$HOME/.supplyzpro.env" "$APP/.env"
cd "$APP"

DOCKER=docker
command -v docker >/dev/null 2>&1 && ! docker ps >/dev/null 2>&1 && DOCKER="sudo docker"
if command -v docker >/dev/null 2>&1 && $DOCKER compose version >/dev/null 2>&1; then
  echo "== deploying with docker compose ($DOCKER)"
  touch mlflow.db && mkdir -p mlruns
  $DOCKER compose down >/dev/null 2>&1 || true
  $DOCKER compose up -d --build
  $DOCKER compose ps
else
  echo "== deploying with a Python virtualenv"
  command -v python3 >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-venv; }
  python3 -m venv .venv 2>/dev/null || { sudo apt-get install -y -qq python3-venv && python3 -m venv .venv; }
  . .venv/bin/activate
  pip install -q --upgrade pip && pip install -q -r requirements.txt
  pkill -f "streamlit run dashboard/app.py" || true
  pkill -f "uvicorn app.api.main:app" || true
  pkill -f "mlflow ui" || true
  export MLFLOW_DISABLE_AGENT_HINT=1
  nohup streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true > streamlit.log 2>&1 &
  nohup uvicorn app.api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
  nohup mlflow ui --backend-store-uri "sqlite:///$APP/mlflow.db" --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
fi

for port in 8501 8000 5000; do
  for i in $(seq 1 60); do curl -s -o /dev/null "http://localhost:$port" && break; sleep 2; done
  printf "port %s: %s\n" "$port" "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$port)"
done
curl -s http://localhost:8000/health || true
