---
title: FinSight AI Dashboard
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.36.0"
app_file: app/frontend/dashboard.py
pinned: true
license: mit
short_description: Explainable Financial AI — Streamlit dashboard
---

# FinSight AI — Dashboard

Streamlit frontend for the FinSight AI explainable financial decision support system.

## Required Secrets (Space Settings → Variables and Secrets)

| Secret | Required | Description |
|--------|----------|-------------|
| `FRONTEND_API_BASE` | Yes | Full URL of the API Space, e.g. `https://YOUR_USERNAME-finsight-api.hf.space/api/v1` |
| `FINSIGHT_API_KEY` | If auth enabled | Same value as `API_SECRET_KEY` set on the API Space |

## Features

- Real-time ML signal (XGBoost / LightGBM / Random Forest)
- SHAP feature attribution waterfall chart
- ML + news signal fusion
- Multi-horizon predictions (1d / 7d / 1m / 6m)
- RAG-grounded financial AI chat
- Autonomous agent with tool calling
- 52-week market data visualisation