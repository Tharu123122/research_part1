from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.prediction import HistoryItem


_history: list[HistoryItem] = []


def add_history_item(
    *,
    risk_percentage: float,
    level: str,
    summary: str,
    voice_status: str,
) -> HistoryItem:
    item = HistoryItem(
        id=str(uuid4()),
        date=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        riskPercentage=round(risk_percentage, 2),
        level=level,
        summary=summary,
        voiceStatus=voice_status,
    )
    _history.insert(0, item)
    return item


def get_history() -> list[HistoryItem]:
    return _history
