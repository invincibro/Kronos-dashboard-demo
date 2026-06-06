"""Background scheduler that runs predictions every hour."""

import threading
import time
from datetime import datetime, timezone

from data_store import update_local_data
from prediction_engine import run_prediction
from storage import save_prediction


class PredictionScheduler:
    def __init__(
        self,
        predictor,
        history_dir,
        pred_len=120,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        interval_seconds=3600,
    ):
        self.predictor = predictor
        self.history_dir = history_dir
        self.pred_len = pred_len
        self.temperature = temperature
        self.top_p = top_p
        self.sample_count = sample_count
        self.interval_seconds = interval_seconds

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

        self.last_run = None
        self.next_run = None
        self.prediction_count = 0
        self.last_error = None
        self.last_data = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def get_status(self):
        with self._lock:
            return {
                "last_run": self.last_run.isoformat() if self.last_run else None,
                "next_run": self.next_run.isoformat() if self.next_run else None,
                "prediction_count": self.prediction_count,
                "last_error": self.last_error,
            }

    def get_latest_data(self):
        with self._lock:
            return self.last_data

    def trigger_now(self):
        """Run a prediction cycle immediately (called from a Flask route)."""
        self._run_cycle()

    def _run_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                self.next_run = datetime.now(timezone.utc) + self._get_interval_td()

            self._run_cycle()

            # Sleep in small increments so we can respond to stop quickly
            interval = self.interval_seconds
            for _ in range(interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _run_cycle(self):
        try:
            df = update_local_data()

            with self._lock:
                self.last_data = df

            pred_df = run_prediction(
                self.predictor,
                df,
                pred_len=self.pred_len,
                temperature=self.temperature,
                top_p=self.top_p,
                sample_count=self.sample_count,
            )

            now = datetime.now(timezone.utc)
            save_prediction(pred_df, now, self.history_dir)

            with self._lock:
                self.last_run = now
                self.prediction_count += 1
                self.last_error = None

        except Exception as e:
            with self._lock:
                self.last_error = str(e)

    def _get_interval_td(self):
        import datetime as _dt

        return _dt.timedelta(seconds=self.interval_seconds)
