import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.core.config import settings
from app.schemas.prediction import HistoryItem, MultimodalJsonRequest, PredictionRecord, PredictionResponse, RiskFactorsRequest, VoicePredictionResponse
from app.services.database_service import close_database, connect_database, database_status, get_history, get_prediction_records, save_prediction_record
from app.services.risk_service import build_structured_prediction, combine_predictions
from app.services.voice_service import predict_uploaded_voice


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await connect_database()
    except Exception as exc:
        print(f"MongoDB connection unavailable: {exc}")
    yield
    await close_database()


app = FastAPI(title="Oral Cancer Risk Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(_: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})


def _history_summary(response: PredictionResponse) -> str:
    if response.finalScore < 35:
        return "Lifestyle markers mildly elevated" if response.finalScore >= 20 else "No severe symptom cluster"
    if response.finalScore < 65:
        return "Follow-up screening support suggested"
    return "Professional assessment strongly recommended"


def _voice_status(response: PredictionResponse) -> str:
    if response.voiceAnalysis is None:
        return "Unavailable"
    return response.voiceAnalysis.mfccPattern


def _build_history_item(response: PredictionResponse) -> HistoryItem:
    from datetime import datetime, timezone
    from uuid import uuid4

    return HistoryItem(
        id=str(uuid4()),
        date=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        riskPercentage=response.riskPercentage,
        level=response.level,
        summary=_history_summary(response),
        voiceStatus=_voice_status(response),
    )


def _parse_risk_factors(value: Any) -> RiskFactorsRequest:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return RiskFactorsRequest.model_validate(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="riskFactors must be valid JSON.") from exc
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.errors()) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name, "database": await database_status()}


@app.post("/api/predict/risk-factors", response_model=PredictionResponse)
async def predict_risk_factors(payload: RiskFactorsRequest) -> PredictionResponse:
    response = build_structured_prediction(payload)
    history_item = _build_history_item(response)
    await save_prediction_record(
        prediction_type="risk_factors",
        request_data=payload.model_dump(mode="json"),
        response_data=response.model_dump(mode="json"),
        history=history_item,
    )
    return response


@app.post("/api/predict/voice", response_model=VoicePredictionResponse)
async def predict_voice(file: UploadFile) -> VoicePredictionResponse:
    file_metadata = {"filename": file.filename, "contentType": file.content_type}
    response = await predict_uploaded_voice(file)
    await save_prediction_record(
        prediction_type="voice",
        request_data={"file": file_metadata},
        response_data=response.model_dump(mode="json"),
        history=None,
    )
    return response


@app.post("/api/predict/multimodal", response_model=PredictionResponse)
async def predict_multimodal(request: Request) -> PredictionResponse:
    content_type = request.headers.get("content-type", "").lower()
    file: UploadFile | None = None
    voice_score: float | None = None
    voice_analysis = None

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        if "riskFactors" not in form:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Multipart requests require a riskFactors JSON field.")
        risk_factors = _parse_risk_factors(form["riskFactors"])
        request_data: dict[str, Any] = {"riskFactors": risk_factors.model_dump(mode="json")}

        uploaded = form.get("file")
        if isinstance(uploaded, (UploadFile, StarletteUploadFile)):
            file = uploaded
            request_data["file"] = {"filename": uploaded.filename, "contentType": uploaded.content_type}

        raw_voice_score = form.get("voiceScore")
        if raw_voice_score not in (None, ""):
            try:
                voice_score = float(str(raw_voice_score))
                request_data["voiceScore"] = voice_score
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="voiceScore must be a number between 0 and 100.") from exc
    else:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must be valid JSON.") from exc
        payload = MultimodalJsonRequest.model_validate(body)
        risk_factors = payload.riskFactors
        voice_score = payload.voiceScore
        request_data = payload.model_dump(mode="json")

    if voice_score is not None and not 0 <= voice_score <= 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="voiceScore must be between 0 and 100.")

    structured = build_structured_prediction(risk_factors)

    if file is not None:
        voice_result = await predict_uploaded_voice(file)
        voice_score = voice_result.voiceScore
        voice_analysis = voice_result.voiceAnalysis

    response = combine_predictions(structured=structured, voice_score=voice_score, voice_analysis=voice_analysis)
    history_item = _build_history_item(response)
    await save_prediction_record(
        prediction_type="multimodal",
        request_data=request_data,
        response_data=response.model_dump(mode="json"),
        history=history_item,
    )
    return response


@app.get("/api/history", response_model=list[HistoryItem])
async def history(limit: int = Query(default=50, ge=1, le=200)) -> list[HistoryItem]:
    return await get_history(limit=limit)


@app.get("/api/predictions", response_model=list[PredictionRecord])
async def predictions(limit: int = Query(default=50, ge=1, le=200)) -> list[PredictionRecord]:
    return await get_prediction_records(limit=limit)
