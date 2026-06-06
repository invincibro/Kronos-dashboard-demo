"""Load Kronos-base model and run predictions."""

import sys
import os
import pandas as pd

# Ensure parent directory is on path so we can import model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Kronos, KronosTokenizer, KronosPredictor

MODEL_CONFIG = {
    "name": "Kronos-base",
    "model_id": "NeoQuasar/Kronos-base",
    "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
    "context_length": 512,
}


def load_kronos_base():
    """Load Kronos-base tokenizer, model, and predictor."""
    tokenizer = KronosTokenizer.from_pretrained(MODEL_CONFIG["tokenizer_id"])
    model = Kronos.from_pretrained(MODEL_CONFIG["model_id"])
    predictor = KronosPredictor(
        model, tokenizer, max_context=MODEL_CONFIG["context_length"]
    )
    return tokenizer, model, predictor


def run_prediction(
    predictor, df, pred_len=120, temperature=1.0, top_p=0.9, sample_count=1
):
    """Run a prediction using the last context_length rows as input.

    Args:
        predictor: KronosPredictor instance
        df: DataFrame with columns [timestamps, open, high, low, close, volume, amount]
        pred_len: Number of periods to predict
        temperature, top_p, sample_count: Sampling parameters

    Returns:
        pred_df: DataFrame with predicted OHLCV+amount indexed by future timestamps
    """
    context_length = MODEL_CONFIG["context_length"]

    if len(df) < context_length:
        raise ValueError(
            f"Need at least {context_length} rows for context, got {len(df)}"
        )

    x_df = df.iloc[-context_length:]
    x_timestamp = x_df["timestamps"]
    if isinstance(x_timestamp, pd.DatetimeIndex):
        x_timestamp = pd.Series(x_timestamp, name="timestamps")

    # Generate future timestamps at the same frequency
    freq = _detect_freq(df["timestamps"])
    last_ts = x_timestamp.iloc[-1]
    y_timestamp = pd.Series(
        pd.date_range(start=last_ts + freq, periods=pred_len, freq=freq),
        name="timestamps",
    )

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
    )

    return pred_df


def _detect_freq(timestamps):
    """Detect the frequency of a datetime Series."""
    if len(timestamps) < 2:
        return pd.Timedelta(minutes=15)
    delta = timestamps.iloc[1] - timestamps.iloc[0]
    return pd.Timedelta(delta)
