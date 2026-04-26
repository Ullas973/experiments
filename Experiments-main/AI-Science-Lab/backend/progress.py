from __future__ import annotations

from db import log_activity


def log_query(user_id: int, query: str) -> None:
    log_activity(user_id=user_id, action_type="query", content=query)


def log_lab(user_id: int, topic: str) -> None:
    log_activity(user_id=user_id, action_type="lab", content=topic)


def log_quiz(user_id: int, topic: str) -> None:
    log_activity(user_id=user_id, action_type="quiz", content=topic)
