import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.core.config import settings
from app.schemas.prediction import HistoryItem, MultimodalJsonRequest, PredictionResponse, RiskFactorsRequest, VoicePredictionResponse
from app.services.report_service import add_history_item, get_history
from app.services.risk_service import build_structured_prediction, combine_predictions
from app.services.voice_service import predict_uploaded_voice


app = FastAPI(title="Oral Cancer Risk Prediction API", version="1.0.0")

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


def _record_prediction(response: PredictionResponse) -> None:
    add_history_item(
        risk_percentage=response.riskPercentage,
        level=response.level,
        summary=_history_summary(response),
        voice_status=_voice_status(response),
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
    return {"status": "ok", "service": settings.service_name}


@app.post("/api/predict/risk-factors", response_model=PredictionResponse)
async def predict_risk_factors(payload: RiskFactorsRequest) -> PredictionResponse:
    response = build_structured_prediction(payload)
    _record_prediction(response)
    return response


@app.post("/api/predict/voice", response_model=VoicePredictionResponse)
async def predict_voice(file: UploadFile) -> VoicePredictionResponse:
    return await predict_uploaded_voice(file)


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

        uploaded = form.get("file")
        if isinstance(uploaded, (UploadFile, StarletteUploadFile)):
            file = uploaded

        raw_voice_score = form.get("voiceScore")
        if raw_voice_score not in (None, ""):
            try:
                voice_score = float(str(raw_voice_score))
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

    if voice_score is not None and not 0 <= voice_score <= 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="voiceScore must be between 0 and 100.")

    structured = build_structured_prediction(risk_factors)

    if file is not None:
        voice_result = await predict_uploaded_voice(file)
        voice_score = voice_result.voiceScore
        voice_analysis = voice_result.voiceAnalysis

    response = combine_predictions(structured=structured, voice_score=voice_score, voice_analysis=voice_analysis)
    _record_prediction(response)
    return response


@app.get("/api/history", response_model=list[HistoryItem])
async def history() -> list[HistoryItem]:
    return get_history()
