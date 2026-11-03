"""Nutrify AI — Production Analytics and Prediction Logging."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger("nutrify.analytics")

_backend_dir = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _backend_dir.parent
REPORTS_DIR = PROJECT_ROOT / "ml" / "data" / "reports"
LOGS_FILE = REPORTS_DIR / "prediction_logs.jsonl"
SUMMARY_FILE = REPORTS_DIR / "prediction_logs.json"


@dataclass
class PredictionLogEntry:
    timestamp: float
    request_id: str
    client_ip: str
    food_name: str
    confidence: float
    confidence_tier: str
    status: str
    latency_ms: float
    is_confident: bool


class AnalyticsTracker:
    """Thread-safe analytics and prediction logger for production monitoring."""

    def __init__(self, reports_dir: Path = REPORTS_DIR) -> None:
        self.reports_dir = reports_dir
        self.logs_file = reports_dir / "prediction_logs.jsonl"
        self.summary_file = reports_dir / "prediction_logs.json"
        self._lock = RLock()

        self._total_predictions: int = 0
        self._confidence_sum: float = 0.0
        self._failed_requests: int = 0
        self._low_confidence_count: int = 0
        self._latencies: list[float] = []

        self._ensure_storage()

    def _ensure_storage(self) -> None:
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            if not self.summary_file.exists():
                self._save_summary()
        except Exception as exc:
            logger.warning(f"Could not initialize analytics storage: {exc}")

    def log_prediction(
        self,
        request_id: str,
        client_ip: str,
        food_name: str,
        confidence: float,
        confidence_tier: str,
        status: str,
        latency_ms: float,
        is_confident: bool,
    ) -> None:
        entry = PredictionLogEntry(
            timestamp=time.time(),
            request_id=request_id,
            client_ip=client_ip,
            food_name=food_name,
            confidence=round(confidence, 4),
            confidence_tier=confidence_tier,
            status=status,
            latency_ms=round(latency_ms, 2),
            is_confident=is_confident,
        )

        with self._lock:
            self._total_predictions += 1
            self._confidence_sum += confidence
            if not is_confident or confidence_tier == "low":
                self._low_confidence_count += 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > 500:
                self._latencies = self._latencies[-500:]

            # Append to jsonl
            try:
                with open(self.logs_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry)) + "\n")
            except Exception as e:
                logger.error(f"Failed to append prediction log: {e}")

            # Periodically update summary file
            if self._total_predictions % 5 == 0 or self._total_predictions <= 10:
                self._save_summary()

    def log_failure(self, request_id: str, client_ip: str, error_code: str) -> None:
        with self._lock:
            self._failed_requests += 1

    def _save_summary(self) -> None:
        metrics = self.get_metrics()
        try:
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write analytics summary: {e}")

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            total = self._total_predictions
            avg_conf = round(self._confidence_sum / total, 4) if total > 0 else 0.0
            avg_lat = round(sum(self._latencies) / len(self._latencies), 2) if self._latencies else 0.0
            sorted_lat = sorted(self._latencies) if self._latencies else [0.0]
            p50 = round(sorted_lat[len(sorted_lat) // 2], 2)
            p95 = round(sorted_lat[int(len(sorted_lat) * 0.95)], 2)
            low_conf_rate = round(self._low_confidence_count / total, 4) if total > 0 else 0.0

            return {
                "total_predictions": total,
                "average_confidence": avg_conf,
                "failed_requests": self._failed_requests,
                "low_confidence_count": self._low_confidence_count,
                "low_confidence_rate": low_conf_rate,
                "latency_ms": {
                    "mean": avg_lat,
                    "p50": p50,
                    "p95": p95,
                },
                "status": "healthy" if self._failed_requests < 10 else "degraded",
            }


# Singleton tracker instance
analytics_tracker = AnalyticsTracker()
