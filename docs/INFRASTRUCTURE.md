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
git lfs install
git lfs pull
docker compose build
docker compose run --rm research-runner python -m pytest -q
```

Run the online readiness check, then start the complete core stack:

```bash
docker compose run --rm shadow-037-preflight
docker compose up -d
docker compose ps
docker compose logs -f shadow-runner-037
```

The default stack contains Spot closed candles for the strategy signal,
USD-M Futures closed candles for paper position monitoring, USD-M Futures
depth for the execution gate, and the continuous 037 shadow runner. Spot depth
is diagnostic-only and can be enabled with `--profile diagnostics`.

The depth collector writes daily Parquet files under:

```text
data/microstructure/futures_ws_depth/btcusdt_depth_YYYY-MM-DD.parquet
```

The closed-candle collector writes:

```text
data/live/spot/btcusdt_5m_closed.parquet
data/live/spot/btcusdt_1h_closed.parquet
data/live/futures_usdm/btcusdt_5m_closed.parquet
data/live/futures_usdm/btcusdt_1h_closed.parquet
```

Seed the candle collector once before the first shadow run:

```bash
docker compose run --rm collector-btcusdt-klines python -m src.infra.binance_kline_collector --seed --timeframe 5m
docker compose run --rm collector-btcusdt-futures-klines python -m src.infra.binance_kline_collector --seed --section execution_kline_collector --timeframe 5m
```

The heartbeat files are:

```text
data/microstructure/futures_ws_depth/runtime/btcusdt_futures_collector_health.json
data/live/spot/btcusdt_kline_health.json
data/live/futures_usdm/btcusdt_kline_health.json
reports/shadow_live_037/runtime/shadow_status.json
```

For a one-command Ubuntu 24.04 installation after cloning into `/opt`:

```bash
sudo bash deploy/bootstrap_ubuntu_24_04.sh
```

The script installs Docker Compose v2, Git LFS and chrony, validates public API
connectivity, installs systemd units and starts the paper-only stack.

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
docker compose exec collector-btcusdt-futures-depth \
  python -m src.infra.binance_microstructure_collector \
  --healthcheck \
  --config config/infrastructure_microstructure_futures.yaml \
  --max-age-seconds 20
```

Docker also runs this check automatically. A healthy collector means a recent
snapshot was written and the heartbeat status is `running`.

Ops healthcheck with disk/parquet checks:

```bash
docker compose run --rm ops-monitor python -m src.infra.ops_monitor --healthcheck --config config/infrastructure_microstructure_futures.yaml
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
http://127.0.0.1:8000/infra/futures_microstructure_quality/
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

The report server binds to localhost. Use an SSH tunnel rather than exposing it:

```bash
ssh -L 8000:127.0.0.1:8000 user@SERVER_IP
```

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
timer runs Futures DQ, writes the shadow daily report and ops report, then runs
the rclone backup. The legacy 017 rolling evaluation remains a manual research
job because it uses the diagnostic Spot-depth dataset.

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

## Shadow Live 037

Run the Iteration 037 zero-fee shadow cycle:

```bash
docker compose exec shadow-runner-037 python -m src.paper.shadow_live_037 --healthcheck
```

The runner uses closed 5m candles, reconstructs complete 1h signal candles,
applies the locked 037 candidate, gates entries with the live depth snapshot and
writes append-only paper outputs under:

```text
reports/shadow_live_037/runtime/
```

Key files:

- `shadow_signals.csv`
- `shadow_orders.csv`
- `shadow_fills.csv`
- `shadow_positions.csv`
- `shadow_daily_pnl.csv`
- `execution_costs_by_day.csv`
- `shadow_status.json`
- `shadow_live_report.md`

The shadow service runs a cycle every 30 seconds so 5m adds, take profits and
forced exits are not monitored only once per hour. The hourly timer performs
ops checks; it does not duplicate the shadow process. It remains paper-only and
refuses private key environment variables or `LIVE_TRADING_ENABLED=true`.

## Fundamental Calendar

`config/fundamental_events_live.csv` contains the official scheduled FOMC,
CPI, PPI and US employment releases through December 2026. The shadow runner
fails closed if it cannot find an eligible scheduled event within the next 45
days. Refresh this file from the official Fed and BLS calendars before that
horizon expires. Unexpected news cannot be known before publication; the live
trend-escape guard remains the protection for those events.

## Guardrails

- No API keys are required by the collector.
- No private Binance endpoints are used.
- No live trading code is started by Docker Compose.
- Signal candles are Spot; paper execution candles and depth are USD-M Futures.
- A signal older than 120 seconds is never replayed with a current book.
- Spot/Futures basis above 10 bps blocks new paper entries.
- Shadow live 037 is paper-only and sends no orders.
- The collector seeds a local order book from REST and then tracks public depth
  updates from WebSocket.
- Binance WebSocket connections can be rotated by the server around 24 hours;
  the collector is designed to reconnect and reseed.
