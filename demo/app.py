"""Flask web app for ETH/USDT live prediction with historical tracking."""

import json
import os
import sys
import threading
from datetime import timedelta
import plotly
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template

# Ensure parent directory is on path for model import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_engine import load_kronos_base
from data_store import load_local_data, update_local_data
from scheduler import PredictionScheduler
from storage import dataframe_to_json, load_all_predictions

app = Flask(__name__)

HISTORY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prediction_history"
)

# Globals
predictor = None
scheduler = None
model_loaded = False
status_lock = threading.Lock()


def initialize():
    """Load model and start scheduler. Called on first request or startup."""
    global predictor, scheduler, model_loaded

    with status_lock:
        if model_loaded:
            return
        try:
            print("Loading Kronos-base model from HuggingFace...")
            _, _, predictor = load_kronos_base()
            print("Model loaded.")

            print("Fetching initial Binance data...")
            update_local_data()
            print("Initial data fetched.")

            scheduler = PredictionScheduler(
                predictor=predictor,
                history_dir=HISTORY_DIR,
                pred_len=120,
                temperature=1.0,
                top_p=0.9,
                sample_count=1,
                interval_seconds=3600,
            )
            scheduler.start()
            print("Scheduler started (every 3600s).")

            # Run an initial prediction immediately
            print("Running initial prediction...")
            scheduler.trigger_now()
            print("Initial prediction complete.")

            model_loaded = True
        except Exception as e:
            print(f"Initialization error: {e}")
            raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    status = {
        "model_loaded": model_loaded,
    }
    if scheduler:
        status.update(scheduler.get_status())
    return jsonify(status)


@app.route("/api/prediction-history")
def api_prediction_history():
    records = load_all_predictions(HISTORY_DIR)
    return jsonify({"success": True, "predictions": records})


@app.route("/api/latest-prediction")
def api_latest_prediction():
    records = load_all_predictions(HISTORY_DIR)
    if records:
        return jsonify({"success": True, "prediction": records[-1]})
    return jsonify({"success": False, "error": "No predictions yet"}), 404


@app.route("/api/chart")
def api_chart():
    """Return a Plotly figure JSON with real data, latest prediction, and historical predictions."""
    try:
        df = load_local_data()
        if df.empty:
            df = update_local_data()
        if scheduler:
            scheduler.last_data = df
    except Exception as e:
        return jsonify({"success": False, "error": f"Data fetch failed: {e}"}), 500

    records = load_all_predictions(HISTORY_DIR)
    fig = go.Figure()

    # ---- Trace 1: Real close-price line ----
    fig.add_trace(
        go.Scatter(
            x=df["timestamps"].tolist(),
            y=df["close"].tolist(),
            mode="lines",
            name="Real Data",
            line=dict(color="#0004FF", width=2),
            opacity=0.85,
        )
    )

    # ---- Traces 2+: Historical prediction close-price lines ----
    colors = [
        "#FF8C00",  # dark orange
        "#DC143C",  # crimson
        "#FF69B4",  # hot pink
    ]
    # display the most recent 3 predictions for clarity
    for i, record in enumerate(records[-3:]):
        preds = record["predictions"]
        if not preds:
            continue

        timestamps = [p["timestamp"] for p in preds]
        closes = [p["close"] for p in preds]
        # Extract just the time portion for the label
        raw_label = record.get("pred_time", "")
        if "T" in raw_label:
            time_label = raw_label.split("T")[1][:5]
        else:
            time_label = raw_label[:5]

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=closes,
                mode="lines",
                name=f"Pred {time_label}",
                line=dict(color=color, width=2),
                opacity=0.75,
            )
        )

    # Calculate x-axis range: last 24h historical + all prediction data
    x_start = (df["timestamps"].iloc[-1] - timedelta(hours=24)).isoformat()
    x_end = df["timestamps"].iloc[-1].isoformat()
    for record in records[-3:]:
        preds = record.get("predictions", [])
        if preds:
            last_pred_ts = preds[-1]["timestamp"]
            if last_pred_ts > x_end:
                x_end = last_pred_ts

    # Layout
    fig.update_layout(
        template="plotly_white",
        height=750,
        title="ETH/USDT 5m — Real Data + Historical Predictions",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis_range=[x_start, x_end],
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({"success": True, "chart": chart_json})


@app.route("/api/volume-chart")
def api_volume_chart():
    """Return a Plotly bar chart JSON for volume data with prediction overlay."""
    try:
        df = load_local_data()
        if df.empty:
            df = update_local_data()
        if scheduler:
            scheduler.last_data = df
    except Exception as e:
        return jsonify({"success": False, "error": f"Data fetch failed: {e}"}), 500

    records = load_all_predictions(HISTORY_DIR)
    fig = go.Figure()

    # Historical volume bars
    fig.add_trace(
        go.Bar(
            x=df["timestamps"].tolist(),
            y=df["volume"].tolist(),
            name="Historical Volume",
            marker_color="#4285F4",
            opacity=0.75,
        )
    )

    # Latest prediction volume bars
    if records:
        latest = records[-1]
        preds = latest["predictions"]
        if preds:
            pred_timestamps = [p["timestamp"] for p in preds]
            pred_volumes = [p["volume"] for p in preds]

            raw_label = latest.get("pred_time", "")

            time_label = raw_label.split("T")[1][:5]

            fig.add_trace(
                go.Bar(
                    x=pred_timestamps,
                    y=pred_volumes,
                    name=f"Pred Volume {time_label}",
                    marker_color="#FF8C00",
                    opacity=0.85,
                )
            )

    # Calculate x-axis range: last 24h historical + all prediction data
    x_start = (df["timestamps"].iloc[-1] - timedelta(hours=24)).isoformat()
    x_end = df["timestamps"].iloc[-1].isoformat()
    if records:
        preds = records[-1].get("predictions", [])
        if preds and preds[-1]["timestamp"] > x_end:
            x_end = preds[-1]["timestamp"]

    fig.update_layout(
        template="plotly_white",
        height=450,
        title="ETH/USDT 5m — Volume",
        xaxis_title="Time",
        yaxis_title="Volume (ETH)",
        xaxis_range=[x_start, x_end],
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        barmode="overlay",
    )

    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({"success": True, "chart": chart_json})


@app.route("/api/amount-chart")
def api_amount_chart():
    """Return a Plotly bar chart JSON for amount data with prediction overlay."""
    try:
        df = load_local_data()
        if df.empty:
            df = update_local_data()
        if scheduler:
            scheduler.last_data = df
    except Exception as e:
        return jsonify({"success": False, "error": f"Data fetch failed: {e}"}), 500

    records = load_all_predictions(HISTORY_DIR)
    fig = go.Figure()

    # Historical amount bars
    fig.add_trace(
        go.Bar(
            x=df["timestamps"].tolist(),
            y=df["amount"].tolist(),
            name="Historical Amount",
            marker_color="#34A853",
            opacity=0.75,
        )
    )

    # Latest prediction amount bars
    if records:
        latest = records[-1]
        preds = latest["predictions"]
        if preds:
            pred_timestamps = [p["timestamp"] for p in preds]
            pred_amounts = [p["amount"] for p in preds]

            raw_label = latest.get("pred_time", "")
            time_label = raw_label.split("T")[1][:5]

            fig.add_trace(
                go.Bar(
                    x=pred_timestamps,
                    y=pred_amounts,
                    name=f"Pred Amount {time_label}",
                    marker_color="#EA4335",
                    opacity=0.85,
                )
            )

    # Calculate x-axis range: last 24h historical + all prediction data
    x_start = (df["timestamps"].iloc[-1] - timedelta(hours=24)).isoformat()
    x_end = df["timestamps"].iloc[-1].isoformat()
    if records:
        preds = records[-1].get("predictions", [])
        if preds and preds[-1]["timestamp"] > x_end:
            x_end = preds[-1]["timestamp"]

    fig.update_layout(
        template="plotly_white",
        height=450,
        title="ETH/USDT 5m — Amount",
        xaxis_title="Time",
        yaxis_title="Amount (USDT)",
        xaxis_range=[x_start, x_end],
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        barmode="overlay",
    )

    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({"success": True, "chart": chart_json})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

os.makedirs(HISTORY_DIR, exist_ok=True)
initialize()

if __name__ == "__main__":
    print("=" * 50)
    print("Kronos ETH/USDT Live Prediction Demo")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=7071)
