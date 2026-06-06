"""Save and load prediction history as JSON files."""

import json
import os
import glob


def save_prediction(pred_df, pred_time, history_dir):
    """Save a prediction record to a JSON file.

    Args:
        pred_df: DataFrame from KronosPredictor.predict() with timestamp index
        pred_time: datetime when the prediction was made
        history_dir: directory to save into

    Returns:
        filename of the saved record
    """
    os.makedirs(history_dir, exist_ok=True)

    predictions = []
    for ts, row in pred_df.iterrows():
        predictions.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "amount": float(row.get("amount", 0)),
            }
        )

    record = {
        "pred_time": (
            pred_time.isoformat() if hasattr(pred_time, "isoformat") else str(pred_time)
        ),
        "pred_len": len(predictions),
        "predictions": predictions,
    }

    filename = f"pred_{pred_time.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(history_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return filename


def load_all_predictions(history_dir):
    """Load all prediction records from the history directory, sorted by pred_time."""
    records = []
    pattern = os.path.join(history_dir, "pred_*.json")

    for filepath in sorted(glob.glob(pattern)):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                record = json.load(f)
            records.append(record)
        except (json.JSONDecodeError, KeyError):
            continue

    return records


def dataframe_to_json(df):
    """Convert a Kronos-format DataFrame to a JSON-serializable list of dicts."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "timestamp": (
                    row["timestamps"].isoformat()
                    if hasattr(row["timestamps"], "isoformat")
                    else str(row["timestamps"])
                ),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "amount": float(row.get("amount", 0)),
            }
        )
    return rows
