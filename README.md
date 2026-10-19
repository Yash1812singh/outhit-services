# Outhit services

Outhit's backend, split into seven services. The website only talks to the **gateway**; the others sit behind it.

```
browser ──► gateway :8001 ──┬──► market-data  :8201   stocks, sectors, crypto, world markets, ML predictions
            login/sign-up   ├──► intelligence :8202   live prices (SSE), news, sentiment, reputation, outlooks, track record
            JWT, rate limit ├──► research     :8203   picks, rockets & growers, F&O, funds, allocation, regime, portfolio
            health          ├──► advisor      :8204   financial planner, profiles, AI chat (Groq, Claude fallback)
                            ├──► paper        :8205   paper portfolios, orders at live prices, copy-trading
                            └──► broker       :8206   real orders, funds & holdings via your own Upstox account
```

| Service | Owns | Database | Calls |
|---|---|---|---|
| gateway | users, passwords, JWTs, routing, rate limits | `gateway/data/auth.db` | all |
| market-data | `/api/market`, `/api/stocks`, `/api/global`, `/api/predict`, `/api/crypto` | `market_data/data/market_data.db` | Yahoo Finance |
| intelligence | `/api/ai/*` (stream, quotes, outlooks, sentiment, model, track-record) | `intelligence/data/intelligence.db` | market-data, news feeds, Groq |
| research | `/api/ai/research`, `picks`, `picks-track-record`, `funds`, `allocation`, `regime`, `portfolio`, `rockets`, `strategies`, `backtest`, `gurus` | `research/data/research.db` | market-data, intelligence, SEC EDGAR, OpenFIGI |
| advisor | `/api/advisor/*` | `advisor/data/advisor.db` | market-data, intelligence, research |
| paper | `/api/paper/*` | `paper/data/paper.db` | research (strategy targets), intelligence (live prices) |
| broker | `/api/broker/*` | `broker/data/broker.db` (secrets encrypted) | Upstox API, intelligence (prices) |

**Trust model.** Only the gateway checks passwords and tokens. It strips any `X-User-Id` / `X-Internal-Token`
the client sends, then forwards the verified user id with the shared `INTERNAL_TOKEN`. Services reject
user-specific and `/internal/*` requests that lack that token. In Docker only the gateway port is published.

Each service has its own database and can be restarted, scaled or redeployed on its own. If one is down,
the gateway returns a 503 for its routes only; the rest of the site keeps working. `GET /api/health` shows
the status of each service.

## Run locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # set INTERNAL_TOKEN, GATEWAY_JWT_SECRET, Groq key
./run.sh                        # starts all seven (honcho); Ctrl+C stops them
./stop.sh                       # frees the ports if something was left running
```

Frontend: `NEXT_PUBLIC_API_URL=http://localhost:8001` (the AI URL defaults to the same gateway).

Run one service alone (e.g. while working on it):

```bash
cd research && PYTHONPATH=.. ../.venv/bin/uvicorn research.main:app --port 8203 --reload
```

## Strategies, backtests and paper trading

- `research/strategies.py`: rule sets in the style of famous investors (Buffett, Greenblatt, Graham, Lynch, O'Neil,
  Minervini, Darvas, Piotroski, Asness, van Vliet), classic portfolios (Dalio, Bogle, Antonacci, Faber, Browne) and a
  custom builder. Each is a function of data up to the signal date, so the backtest, today's holdings and copy-trading
  all run the same code.
- `research/gurus.py`: real portfolios from 13F filings (Berkshire, Pershing Square, Himalaya, Baupost, Duquesne,
  Appaloosa, Dalal Street, Fundsmith), copied the day after each filing became public.
- `research/backtest.py`: trades the day after each signal, with costs. Stock strategies are judged against an
  equal-weighted basket of the same companies, because the universe is today's largest (survivorship bias).
- `paper/`: virtual portfolios. Manual ones take market orders at live prices, with fees. Copy ones ask research
  for `/internal/strategies/{key}/targets` on the strategy's schedule (or on each new 13F) and trade to match.
  Paper results are the unbiased, forward-looking test of a strategy.
- Set `RESEARCH_SEC_USER_AGENT` to "AppName contact@yourdomain" (SEC asks automated clients to identify themselves).

## Real trading (broker service, Upstox)

Users connect their own Upstox account; Outhit never holds money.

- **Test mode** (default): paste a token from the Sandbox tab of https://account.upstox.com/developer/apps. Orders go to
  `api-sandbox.upstox.com` (Upstox's sandbox supports place/modify/cancel only, so funds and holdings need live mode).
- **Live mode**: create an Upstox app whose Redirect URL is `BROKER_REDIRECT_URI` (default
  `http://localhost:3050/trade/broker/callback`), enter its key and secret, and log in on Upstox (OAuth, state-checked).
  Upstox logins expire daily at 3:30 AM IST.
- Safety:
  - Live orders also need the "live trading" switch, which you turn on by typing a confirmation.
  - Every order is previewed, then confirmed once; the preview is single-use and expires after 2 minutes.
  - Per-order and per-day value limits (capped server-side), a fat-finger price check and a pause switch are checked
    again at confirm time.
  - API secrets and tokens are encrypted with `BROKER_ENCRYPTION_KEY` (Fernet) and never returned to the browser.
- Funds: cash and margin, and deposit/withdrawal history, come from Upstox. Add/Withdraw open Upstox itself.

## Docker

```bash
docker compose up --build       # gateway on http://localhost:8001
```

One `Dockerfile` builds every service (`--build-arg SERVICE=...`). Each service has its own `requirements.txt`
and a named volume for its data.

## One process (small hosts)

`combined.py` runs all seven services in one process: the gateway at the root, the others under
`/_svc/<name>` (only reachable with `INTERNAL_TOKEN`), each still with its own database in `OUTHIT_DATA_DIR`. Point the host's health check at `/healthz`.

```bash
set -a; source .env; set +a
OUTHIT_DATA_DIR=./data .venv/bin/uvicorn combined:app --port 8001
```

It needs roughly 300 MB idle and 450 MB with the background jobs running; training predictions and the
research jobs (picks, rockets, backtests) add 200–400 MB more. Settings for small or public servers:

| Env var | Default | |
|---|---|---|
| `BROKER_ALLOW_LIVE` | `true` | `false`: test mode (Upstox sandbox) only, no real-money orders |
| `MARKET_DATA_ENABLE_TRAINING` | `true` | `false`: the Predict page says predictions aren't available here |
| `MARKET_DATA_PREDICT_UNIVERSE` | `300` | Stocks each prediction model learns from |
| `INTELLIGENCE_DISABLED_LOOPS` | empty | Background loops to skip, e.g. `news,outlooks,llm-news,ledger,bootstrap` (each sweeps every tracked company) |
| `RESEARCH_DISABLED_JOBS` | empty | Research jobs to switch off by name prefix, e.g. `tournament,rockets,trade:bt,rec:,intraday` |
| `OUTHIT_BACKUP_REPO` + `HF_TOKEN` | unset | Back up the databases to a private Hugging Face dataset (for hosts that wipe the disk) |

## Tests

```bash
.venv/bin/python -m pytest
```

## Migrating from the old monolith

`scripts/migrate_legacy.py` copies users, profiles, quotes, fundamentals, news, predictions and the picks log
from `outhit-backend/outhit.db` and `outhit-ai/outhit_ai.db`. It is safe to re-run. If you keep the same JWT
secret, existing logins stay valid.
