# Grid Survival Research Infrastructure

This infrastructure is research-only. It collects public Binance market data,
runs local backtests/research jobs, and never places orders or uses private API
keys.

## Recommended VPS

- Region: Europe close to Binance routes, e.g. Frankfurt, Amsterdam, Paris.
- Size: 2 vCPU, 4 GB RAM, 80-160 GB NVMe.
- OS: Ubuntu 24.04 LTS.
- Runtime: Docker Engine with Compose v2.

## First Deploy

```bash
sudo mkdir -p /opt/grid-survival-research
sudo chown "$USER:$USER" /opt/grid-survival-research
git clone https://github.com/Pavdot/grid-survival-research.git /opt/grid-survival-research
cd /opt/grid-survival-research
docker compose build
docker compose run --rm research-runner python -m pytest -q
```

Start the BTCUSDT public market-data collector:

```bash
docker compose up -d collector-btcusdt-depth
docker compose logs -f collector-btcusdt-depth
```

The collector writes daily Parquet files under:

```text
data/microstructure/ws_depth/btcusdt_depth_YYYY-MM-DD.parquet
```

The heartbeat file is:

```text
data/microstructure/ws_depth/btcusdt_collector_health.json
```

## Systemd Autostart

Copy the service template and enable it:

```bash
sudo cp deploy/systemd/grid-survival-research.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable grid-survival-research
sudo systemctl start grid-survival-research
sudo systemctl status grid-survival-research
```

The service expects the repo at `/opt/grid-survival-research`. Edit
`WorkingDirectory` in the service file if you deploy elsewhere.

## Healthcheck

Manual healthcheck:

```bash
docker compose exec collector-btcusdt-depth \
  python -m src.infra.binance_microstructure_collector \
  --healthcheck \
  --config config/infrastructure_microstructure.yaml \
  --max-age-seconds 20
```

Docker also runs this check automatically. A healthy collector means a recent
snapshot was written and the heartbeat status is `running`.

Ops healthcheck with disk/parquet checks:

```bash
docker compose run --rm ops-monitor python -m src.infra.ops_monitor --healthcheck
```

Telegram test alert:

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
docker compose run --rm ops-monitor python -m src.infra.ops_monitor --send-test-alert
```

## Research Evaluation

After collecting at least 24 hours, run:

```bash
docker compose run --rm research-runner \
  python -m src.research.microstructure_order_policy_017 --evaluate-policies
```

For a quick smoke check:

```bash
docker compose run --rm research-runner \
  python -m src.research.microstructure_order_policy_017 --smoke --max-snapshots 100 --max-trades 50
```

Outputs are written under:

```text
reports/research_iterations/iteration_022_microstructure_order_policy_017/
```

Rolling execution evaluation from WebSocket depth files:

```bash
docker compose run --rm daily-evaluation
```

Outputs are written under:

```text
reports/research_iterations/iteration_026_rolling_execution_evaluation/
```

The 026 verdict uses rolling 24h / 7d / 30d execution metrics and keeps the
Iteration 017 strategy locked. It never re-selects strategy candidates.

## Data Quality Dashboard

Generate the static dashboard:

```bash
docker compose run --rm quality-dashboard
```

Serve reports:

```bash
docker compose --profile manual up -d report-server
```

Open:

```text
http://SERVER_IP:8000/infra/microstructure_quality/
```

The DQ module reports expected versus received snapshots, time gaps, stale
periods, spread/depth/imbalance, latency, crossed books and zero depth. Scores:
`healthy`, `degraded`, or `bad`.

## Reports

Start a local report server:

```bash
docker compose --profile manual up -d report-server
```

Then open:

```text
http://SERVER_IP:8000/
```

If the VPS is public, firewall port `8000` or tunnel it over SSH instead of
exposing it to the internet.

## Backups

Minimum daily backup targets:

- `data/microstructure/ws_depth/*.parquet`
- `data/processed/*.parquet`
- `reports/research_iterations/*`

Production backups use `rclone` and an S3-compatible remote configured outside
Git:

```bash
sudo tee /etc/grid-survival-research.env >/dev/null <<'EOF'
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
RCLONE_REMOTE=gridbackup:grid-survival-research
LIVE_TRADING_ENABLED=false
EOF
sudo chmod 600 /etc/grid-survival-research.env
```

Dry-run:

```bash
docker compose run --rm ops-monitor python -m src.infra.ops_monitor --backup-dry-run
```

Real backup:

```bash
docker compose run --rm ops-monitor python -m src.infra.ops_monitor --backup
```

Backup manifests are written under:

```text
reports/infra/backups/
```

## Timers

Install collector autostart and scheduled jobs:

```bash
sudo cp deploy/systemd/grid-survival-research.service /etc/systemd/system/
sudo cp deploy/systemd/grid-survival-ops-hourly.service /etc/systemd/system/
sudo cp deploy/systemd/grid-survival-ops-hourly.timer /etc/systemd/system/
sudo cp deploy/systemd/grid-survival-daily.service /etc/systemd/system/
sudo cp deploy/systemd/grid-survival-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now grid-survival-research
sudo systemctl enable --now grid-survival-ops-hourly.timer
sudo systemctl enable --now grid-survival-daily.timer
```

The hourly timer checks collector/parquet freshness and disk usage. The daily
timer runs DQ, rolling execution evaluation, daily ops report and rclone backup.

## Paper-Only Harness

Run one paper-only cycle:

```bash
docker compose run --rm paper-runner
```

The harness refuses to start if `LIVE_TRADING_ENABLED=true` or if Binance
private API key environment variables are present. Outputs are written under:

```text
reports/paper_trading/
```

Iteration 027 remains simulation only: no private endpoints, no account access
and no real orders.

## Guardrails

- No API keys are required by the collector.
- No private Binance endpoints are used.
- No live trading code is started by Docker Compose.
- The collector seeds a local order book from REST and then tracks public depth
  updates from WebSocket.
- Binance WebSocket connections can be rotated by the server around 24 hours;
  the collector is designed to reconnect and reseed.
