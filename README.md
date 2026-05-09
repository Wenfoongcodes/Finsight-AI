# 📈 FinSight AI — Explainable Financial Decision Support System

A production-grade AI system combining financial machine learning, explainable AI (SHAP/LIME), RAG-grounded LLM chat, and agentic AI workflows — served through a FastAPI backend and Streamlit frontend.

---

## Architecture Overview

```
finsight-ai/
├── app/
│   ├── core/           # Exceptions, logging
│   ├── api/            # FastAPI schemas, versioned routers
│   ├── ml/             # Data ingestion, feature engineering, models, training, evaluation, explainability
│   ├── rag/            # RAG pipeline (FAISS + sentence-transformers), LLM chat
│   ├── agents/         # Agentic AI orchestrator + tools
│   ├── services/       # PredictionService (orchestrates ML pipeline)
│   └── frontend/       # Streamlit dashboard
├── configs/            # Pydantic settings (env-driven)
├── scripts/            # Offline training and ingestion CLIs
├── tests/              # pytest unit tests
├── requirements/       # Layered pip requirements (base / ml / dev)
├── docker/             # Production Dockerfile (multi-stage)
└── .github/workflows/  # GitHub Actions CI
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/yourname/finsight-ai.git
cd finsight-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/ml.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### 3. Pre-warm data cache (optional but recommended)

```bash
python scripts/ingest_data.py --defaults --years 5
```

### 4. Train a model

```bash
python scripts/train_model.py --ticker AAPL --model xgboost
python scripts/train_model.py --tickers AAPL MSFT GOOGL --model lightgbm --hpo
```

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

# API only
docker-compose up api
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/predict/` | Next-day price direction prediction |
| POST | `/api/v1/predict/batch` | Batch predictions |
| POST | `/api/v1/train/` | Train a model for a ticker |
| POST | `/api/v1/market/summary` | OHLCV summary statistics |
| POST | `/api/v1/rag/ingest` | Add documents to knowledge base |
| POST | `/api/v1/rag/chat` | Conversational AI assistant |
| POST | `/api/v1/agent/run` | Agentic AI query execution |

Full interactive docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ML Pipeline

```
yfinance → OHLCV validation → Feature engineering (RSI, MACD, Bollinger Bands, ATR, OBV, ...)
→ Walk-forward training (XGBoost / LightGBM / RF / LogReg)
→ SHAP explanation → PredictionResponse
```

**Walk-forward validation** is used throughout — no future leakage. Models are evaluated across 5 expanding-window folds.

---

## Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Supported Models

| Model | Registry Key |
|-------|-------------|
| XGBoost | `xgboost` |
| LightGBM | `lightgbm` |
| Random Forest | `random_forest` |
| Logistic Regression | `logistic_regression` |
| LSTM (optional) | requires TensorFlow |

---

## Environment Variables

See `.env.example` for all configurable values. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM features | required |
| `LLM_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `ENVIRONMENT` | Runtime environment | `development` |
| `DEBUG` | Enable debug logging | `false` |

---

## License

MIT
