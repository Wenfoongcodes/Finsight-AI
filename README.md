# 📈 FinSight AI — Explainable Financial Decision Support System

A production-grade AI system combining financial machine learning, explainable AI (SHAP/LIME), RAG-grounded LLM chat, and agentic AI workflows — served through a FastAPI backend and Streamlit frontend.

---

## Architecture Overview

```
finsight-ai/
├── app/
│   ├── core/           # Exceptions, logging, security middleware, formatting contracts, cache cleanup
│   ├── api/            # FastAPI schemas (prediction, portfolio, versioning), versioned routers
│   ├── ml/             # Data ingestion, feature engineering, model factory, training, versioning, evaluation, explainability
│   ├── rag/            # RAG pipeline (FAISS + sentence-transformers), LLM chat
│   ├── agents/         # Agentic AI orchestrator + tools
│   ├── services/       # PredictionService, ModelSelector, SignalFusionService, NewsIntelligenceService,
│   │                    #   PortfolioAnalysisService, TickerResolver
│   └── frontend/       # Streamlit dashboard + portfolio tab
├── configs/            # Pydantic settings (env-driven)
├── scripts/            # Offline training/ingestion CLIs, Redis rate-limit verification, options cache warmer
├── tests/              # pytest unit tests (400+ tests across ingestion, features, training, services, routes)
├── requirements/       # Layered pip requirements (base / ml / dev)
├── docker/             # Production Dockerfile (multi-stage)
├── Dockerfile.hf       # HuggingFace Spaces Dockerfile
└── .github/workflows/  # GitHub Actions CI
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/Wenfoongcodes/Finsight-AI.git
cd Finsight-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/ml.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### 3. Pre-warm data cache (optional but recommended)

Downloads and caches OHLCV data for a default watchlist of 10 tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, GS, SPY).

```bash
# Default watchlist, 5 years
python scripts/ingest_data.py --defaults --years 5

# Single ticker
python scripts/ingest_data.py --ticker AAPL --years 3

# Multiple tickers, bypass cache
python scripts/ingest_data.py --tickers AAPL MSFT GOOGL --no-cache
```

Optionally pre-warm the options/implied-volatility cache:

```bash
python scripts/warm_options_cache.py --tickers AAPL MSFT GOOGL
```

### 4. Train a model

Trains with walk-forward validation, automated feature selection, and persists versioned model artifacts to `data/models/`.

```bash
# Basic training
python scripts/train_model.py --ticker AAPL --model xgboost

# Multiple tickers
python scripts/train_model.py --tickers AAPL MSFT GOOGL --model lightgbm

# With Optuna hyperparameter optimisation
python scripts/train_model.py --ticker AAPL --model xgboost --hpo --hpo-trials 50

# All options
python scripts/train_model.py --ticker NVDA --model random_forest --period-years 3 --hpo
```

Available models: `xgboost`, `lightgbm`, `random_forest`, `logistic_regression`

### 5. Start the API server

```bash
uvicorn main:app --reload --port 8000
```

### 6. Start the dashboard

```bash
streamlit run app/frontend/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker

```bash
# Build and run full stack (API + Dashboard)
docker-compose up --build

# Production stack with Nginx (TLS termination)
docker-compose --profile production up --build

# API only
docker-compose up api
```

See `docker-compose.yml` for the full service configuration. Place TLS certs at `docker/nginx/certs/fullchain.pem` and `docker/nginx/certs/privkey.pem` for production with Nginx.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — reports LLM, auth, and rate-limit status |
| POST | `/api/v1/predict/` | Next-day (or multi-horizon) price direction prediction |
| POST | `/api/v1/predict/batch` | Batch predictions for multiple tickers |
| GET | `/api/v1/predict/leaderboard/{ticker}` | Model performance leaderboard for a ticker |
| POST | `/api/v1/predict/stream` | Server-Sent Events stream of prediction pipeline progress |
| POST | `/api/v1/train/` | Train a model for a ticker/horizon |
| POST | `/api/v1/market/summary` | OHLCV summary statistics |
| POST | `/api/v1/rag/ingest` | Add documents or article URLs to knowledge base |
| POST | `/api/v1/rag/chat` | Conversational AI assistant (RAG-grounded) |
| POST | `/api/v1/agent/run` | Agentic AI query execution |
| POST | `/api/v1/agent/stream` | SSE stream of agent plan → tool-execution → synthesis progress |
| GET | `/api/v1/versions/{ticker}/{model_name}/{horizon}` | Full version history for a model slot, with metrics per version |
| POST | `/api/v1/versions/promote` | Promote a specific model version to active (upgrade or rollback) |
| POST | `/api/v1/versions/rollback` | Roll back to the previously active version |
| POST | `/api/v1/portfolio/analyze` | Portfolio-level analysis: correlation, mean-variance optimization, efficient frontier, risk attribution, VaR |

Full interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ML Pipeline

```
yfinance → OHLCV validation
  → Feature engineering (technical indicators, sector/market correlation, options & IV, fundamental/macro)
  → Automated feature selection (StabilityBasedFeatureSelector)
  → Walk-forward training (XGBoost / LightGBM / RF / LogReg) with Optuna HPO
  → Platt scaling (probability calibration)
  → Versioned artifact persistence (promote / rollback / prune)
  → Leaderboard-based auto-selection
  → SHAP explanation → Signal fusion (ML + news) → PredictionResponse
```

**Walk-forward validation** is used throughout — no future leakage. Models are evaluated across 5 expanding-window folds. The system automatically trains all candidate models on first prediction request if no artifacts exist, then selects the best by AUC.

**Feature engineering** spans technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, ...), sector/market correlation against SPDR ETFs, options market data (ATM implied volatility, IV rank, put/call ratios, VIX term structure), and fundamental/macroeconomic features (valuation ratios, profitability, growth, financial health, earnings surprises, macro context).

**Automated feature selection** runs stability-based selection as a pipeline stage integrated into walk-forward training, with results persisted in versioned artifacts and surfaced in the dashboard as a plain-English transparency panel.

**Model versioning** keeps an immutable artifact store per ticker/model/horizon with an `active.json` pointer, exposed via `/api/v1/versions/*` for promotion, rollback, and history inspection.

**Portfolio analysis** layers mean-variance optimization, an efficient frontier, correlation/risk attribution, sector exposure, and Value-at-Risk on top of per-ticker predictions, dispatched concurrently via a `ThreadPoolExecutor` to bound latency.

**Signal fusion** combines the ML prediction with source-weighted, recency-filtered financial news via an LLM synthesis step. Falls back to a deterministic rule-based fusion if the LLM is unavailable.

---

## Supported Models

| Model | Registry Key | Notes |
|-------|-------------|-------|
| XGBoost | `xgboost` | Default; calibrated with Platt scaling |
| LightGBM | `lightgbm` | Calibrated with Platt scaling |
| Random Forest | `random_forest` | Calibrated with Platt scaling |
| Logistic Regression | `logistic_regression` | Natively calibrated; no Platt scaling |
| LSTM (optional) | — | Requires TensorFlow; see `model_factory.py` |

---

## Prediction Horizons

| Horizon | Key | Trading days |
|---------|-----|-------------|
| Next day | `1d` | 1 |
| Next week | `7d` | 5 |
| Next month | `1m` | 21 |
| Next 6 months | `6m` | 126 |

News recency filtering is horizon-aware: a `1d` prediction only considers articles from the last 3 days; `6m` accepts articles up to 90 days old.

---

## Rate Limiting

Per-IP rate limiting protects the API and can run on two backends, controlled by `RATE_LIMIT_BACKEND`:

- `memory` (default) — in-process sliding window, suitable for single-instance deployments.
- `redis` — distributed sliding-window limiter implemented in Lua, for multi-instance/production deployments. Requires `REDIS_URL` (or host/port) to be configured.

Verify a Redis-backed limiter with:

```bash
python scripts/verify_redis_ratelimit.py
```

---

## Running Tests

```bash
# Full test suite with coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Fast subset
pytest tests/test_feature_engineering.py tests/test_data_ingestion.py -v
```

Tests cover data ingestion, feature engineering (technical, sector/correlation, options, fundamental), walk-forward training and versioning, model loading, portfolio analysis, signal fusion, the RAG pipeline, the agentic orchestrator, rate limiting, and all FastAPI route handlers.

---

## Environment Variables

See `.env.example` for all configurable values.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key — required for LLM features | — |
| `LLM_MODEL` | LLM model name | `gpt-4o-mini` |
| `LLM_BASE_URL` | Override for alternative providers (Groq, Ollama, Azure) | OpenAI |
| `ENVIRONMENT` | Runtime environment (`development` / `production`) | `development` |
| `DEBUG` | Enable debug logging | `false` |
| `API_KEY_ENABLED` | Gate API behind `X-API-Key` header | `false` |
| `API_SECRET_KEY` | Shared API secret (generate with `secrets.token_urlsafe(32)`) | — |
| `RATE_LIMIT_ENABLED` | Enable per-IP rate limiting | `true` |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window | `120` |
| `RATE_LIMIT_WINDOW_S` | Rate limit window in seconds | `60` |
| `RATE_LIMIT_BACKEND` | `memory` or `redis` | `memory` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins, or `*` | `localhost:3000,localhost:8501` |
| `FRONTEND_API_BASE` | API URL used by Streamlit dashboard | `http://localhost:8000/api/v1` |
| `FINSIGHT_API_KEY` | API key injected into dashboard requests | — |
| `MODELS_DIR` | Path for versioned model artifacts | `data/models` |
| `RAW_DATA_DIR` | Path for cached parquet files | `data/raw` |
| `EMBEDDINGS_DIR` | Path for FAISS index and docs | `data/embeddings` |
| `CACHE_MAX_AGE_DAYS` | Max age (days) before data cache is refreshed | `1` |

### Alternative LLM Providers

Set `LLM_BASE_URL` and `LLM_MODEL` together:

```bash
# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b

# Ollama (local)
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
```

---

## HuggingFace Spaces Deployment

The repo includes a `Dockerfile.hf` configured for HF Spaces requirements:
- Listens on port **7860** (the only publicly exposed HF port)
- Runs as **UID 1000** (HF requirement)
- Stores all artifacts on `/data` (persistent volume, survives restarts)

Set the following in Space Settings → Variables and Secrets:

| Secret | Description |
|--------|-------------|
| `OPENAI_API_KEY` | LLM API key |
| `API_SECRET_KEY` | Shared API key for authentication |
| `FRONTEND_API_BASE` | Full URL of the API Space |
| `FINSIGHT_API_KEY` | Same as `API_SECRET_KEY`; used by the dashboard |

---

## License

MIT