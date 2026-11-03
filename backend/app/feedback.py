"""Nutrify AI — User Feedback Collection and Retraining Data Pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("nutrify.feedback")

_backend_dir = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _backend_dir.parent
FEEDBACK_DIR = PROJECT_ROOT / "ml" / "data" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "user_feedback.jsonl"


@dataclass
class UserFeedbackEntry:
    timestamp: float
    request_id: str
    predicted_food: str
    is_correct: bool
    corrected_food: str | None
    comments: str | None
    client_ip: str | None = None


class FeedbackManager:
    """Stores user correction feedback locally to prepare future retraining datasets."""

    def __init__(self, feedback_file: Path = FEEDBACK_FILE) -> None:
        self.feedback_file = feedback_file
        self.feedback_dir = feedback_file.parent
        self._lock = Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            self.feedback_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not create feedback directory: {exc}")

    def record_feedback(
        self,
        request_id: str,
        predicted_food: str,
        is_correct: bool,
        corrected_food: str | None = None,
        comments: str | None = None,
        client_ip: str | None = None,
    ) -> dict[str, Any]:
        entry = UserFeedbackEntry(
            timestamp=time.time(),
            request_id=request_id,
            predicted_food=predicted_food,
            is_correct=is_correct,
            corrected_food=corrected_food if not is_correct else None,
            comments=comments,
            client_ip=client_ip,
        )

        with self._lock:
            try:
                with open(self.feedback_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry)) + "\n")
                logger.info(f"Recorded user feedback for request {request_id}: correct={is_correct}")
                return {"success": True, "message": "Feedback recorded. Thank you for helping improve Nutrition AI!"}
            except Exception as e:
                logger.error(f"Failed to record feedback: {e}")
                return {"success": False, "message": f"Storage error: {e}"}

    def get_feedback_count(self) -> int:
        if not self.feedback_file.exists():
            return 0
        with self._lock:
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
            except Exception:
                return 0


# Singleton manager instance
feedback_manager = FeedbackManager()
