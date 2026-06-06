# Kronos Demo — Setup and Configuration

## Overview

The Kronos demo is a live prediction web app for **ETH/USDT** 15-minute K-lines. It fetches real data from Binance, runs the Kronos-base model on an hourly schedule, and displays historical data alongside past predictions in a Plotly chart.

## Prerequisites

- **Python 3.10+** (or Docker)
- **HuggingFace Hub** access (the model downloads from `NeoQuasar/Kronos-base` and `NeoQuasar/Kronos-Tokenizer-base`) — no token required for public models
- **Internet access** to fetch from Binance's public data API (`data-api.binance.vision`)

## Quick Start (Docker)

```bash
docker compose up --build
```

The app will be available at **http://localhost:9090**.

This mounts two directories for persistence across container restarts:

- `./data/` — cached ETH/USDT CSV data (rolling 30-day window)
- `./prediction_history/` — saved prediction JSON files

## Manual Setup (without Docker)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo

```bash
cd demo
python app.py
```

The app starts on **http://localhost:7071**.

On first launch it will:

1. Download Kronos-base tokenizer and model from HuggingFace Hub (~400 MB)
2. Fetch the last 30 days of ETH/USDT 15m K-lines from Binance
3. Run an initial prediction
4. Start the hourly prediction scheduler

## Configuration

### Prediction parameters

Edit `demo/scheduler.py` to change the defaults. Key settings in the `PredictionScheduler` constructor (line 22 of `demo/app.py`):

| Parameter          | Default | Description                                 |
| ------------------ | ------- | ------------------------------------------- |
| `pred_len`         | 120     | Number of future periods to predict         |
| `temperature`      | 1.0     | Sampling temperature (higher = more varied) |
| `top_p`            | 0.9     | Nucleus sampling threshold                  |
| `sample_count`     | 1       | Number of prediction samples                |
| `interval_seconds` | 3600    | Time between scheduled predictions (1 hour) |

### Data interval

Edit `demo/binance_fetcher.py` to change the K-line interval. The `INTERVAL` constant (line 8) defaults to `"5m"`. Supported values: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`, etc.

The data store (`demo/data_store.py`) maintains a rolling 30-day CSV at `demo/data/ethusdt_15m.csv`.

### Model variant

Edit the `MODEL_CONFIG` dict in `demo/prediction_engine.py` (line 12) to switch models:

```python
# Kronos-mini (4.1M params, 2048 context — fastest)
MODEL_CONFIG = {
    "model_id": "NeoQuasar/Kronos-mini",
    "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
    "context_length": 2048,
}

# Kronos-small (24.7M params, 512 context — balanced)
MODEL_CONFIG = {
    "model_id": "NeoQuasar/Kronos-small",
    "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
    "context_length": 512,
}

# Kronos-base (102.3M params, 512 context — default, highest quality)
MODEL_CONFIG = {
    "model_id": "NeoQuasar/Kronos-base",
    "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
    "context_length": 512,
}
```

### Port mapping (Docker)

Edit `docker-compose.yml` to change the host port:

```yaml
ports:
  - "<host_port>:7071"
```

The container always listens on port 7071 internally.

## API Endpoints

| Endpoint                  | Method | Description                                          |
| ------------------------- | ------ | ---------------------------------------------------- |
| `/`                       | GET    | Main UI with live chart                              |
| `/api/status`             | GET    | Scheduler status (last run, next run, count, errors) |
| `/api/chart`              | GET    | Plotly JSON with real data + last 3 predictions      |
| `/api/latest-prediction`  | GET    | Most recent prediction record                        |
| `/api/prediction-history` | GET    | All saved prediction records                         |

## Files

```
demo/
  app.py                   Flask app entry point
  binance_fetcher.py       Binance REST API client
  data_store.py            Local CSV data cache (rolling 30-day)
  prediction_engine.py     Kronos model loading and prediction
  scheduler.py             Background hourly prediction thread
  storage.py               JSON prediction history I/O
  templates/index.html     Frontend UI
  data/                    Cached ETH/USDT CSV (auto-created)
  prediction_history/      Prediction JSON records (auto-created)
```

## Troubleshooting

- **Model download fails**: Check your internet connection and HuggingFace Hub availability. The models are ~200 MB each.
- **Binance data fetch fails**: The public API at `data-api.binance.vision` may be rate-limited. The scheduler retries each cycle.
- **Out of memory**: Kronos-base requires ~400 MB VRAM on GPU. Use `kronos-mini` (~16 MB) if resources are tight.
- **Port already in use**: Change the port in `docker-compose.yml` (Docker) or the `app.run()` call in `demo/app.py` (manual).
