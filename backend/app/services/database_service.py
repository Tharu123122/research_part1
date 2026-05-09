from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.core.config import settings
from app.schemas.prediction import HistoryItem, PredictionRecord


_client: Any | None = None
_memory_records: list[PredictionRecord] = []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _collection() -> Any | None:
    if _client is None:
        return None
    database = _client[settings.mongodb_database]
    return database[settings.predictions_collection]


def _serialize_record(record: PredictionRecord) -> dict[str, Any]:
    document = record.model_dump(mode="json")
    document["_id"] = document.pop("id")
    return document


def _record_from_document(document: dict[str, Any]) -> PredictionRecord:
    data = dict(document)
    data["id"] = str(data.pop("_id"))
    return PredictionRecord.model_validate(data)


async def connect_database() -> None:
    global _client
    if not settings.mongodb_uri:
        return
    from motor.motor_asyncio import AsyncIOMotorClient

    client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=settings.connection_timeout_ms,
    )
    try:
        await client.admin.command("ping")
    except Exception:
        client.close()
        raise
    _client = client
    collection = _collection()
    if collection is not None:
        await collection.create_index([("date", -1)])
        await collection.create_index([("predictionType", -1), ("date", -1)])


async def close_database() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def database_status() -> str:
    if not settings.mongodb_uri:
        return "not_configured"
    if _client is None:
        return "disconnected"
    try:
        await _client.admin.command("ping")
    except Exception:
        return "unavailable"
    return "connected"


async def save_prediction_record(
    *,
    prediction_type: str,
    request_data: dict[str, Any],
    response_data: dict[str, Any],
    history: HistoryItem | None,
) -> PredictionRecord:
    record = PredictionRecord(
        id=str(uuid4()),
        date=_utc_now(),
        predictionType=prediction_type,
        request=request_data,
        response=response_data,
        history=history,
    )

    collection = _collection()
    if collection is None:
        _memory_records.insert(0, record)
        return record

    try:
        await collection.insert_one(_serialize_record(record))
    except Exception:
        _memory_records.insert(0, record)
    return record


async def get_prediction_records(limit: int = 50) -> list[PredictionRecord]:
    limit = max(1, min(limit, 200))
    collection = _collection()
    if collection is None:
        return _memory_records[:limit]

    try:
        cursor = collection.find({}).sort("date", DESCENDING).limit(limit)
        return [_record_from_document(document) async for document in cursor]
    except Exception:
        return _memory_records[:limit]


async def get_history(limit: int = 50) -> list[HistoryItem]:
    records = await get_prediction_records(limit=limit)
    return [record.history for record in records if record.history is not None]
