import importlib.util
import math
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.prediction import VoiceAnalysis, VoicePredictionResponse
from app.utils.paths import VOICE_FEATURE_SCRIPT_PATH, VOICE_MODEL_PATH

BUNDLED_VOICE_FEATURE_SCRIPT_PATH = Path(__file__).with_name("voice_feature_extractor.py")
BUNDLED_VOICE_MODEL_PATH = Path(__file__).with_name("models") / "voice_abnormality_model.joblib"


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


@lru_cache(maxsize=1)
def _load_voice_feature_module() -> Any:
    script_path = VOICE_FEATURE_SCRIPT_PATH if VOICE_FEATURE_SCRIPT_PATH.exists() else BUNDLED_VOICE_FEATURE_SCRIPT_PATH
    if not script_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Voice feature extraction script is missing.")
    spec = importlib.util.spec_from_file_location("extract_voice_features", script_path)
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Voice feature extractor cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "extract_features"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Voice feature extraction function is unavailable.")
    return module


@lru_cache(maxsize=1)
def _load_voice_model() -> dict[str, Any]:
    model_path = VOICE_MODEL_PATH if VOICE_MODEL_PATH.exists() else BUNDLED_VOICE_MODEL_PATH
    if not model_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Voice abnormality model file is missing.")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload or "feature_columns" not in payload:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Voice model payload is invalid.")
    return payload


def _voice_label(score: float) -> str:
    if score < 35:
        return "stable"
    if score < 65:
        return "slight_variation"
    return "abnormal_marker"


def _voice_analysis(score: float, features: dict[str, Any]) -> VoiceAnalysis:
    jitter = _finite_float(features.get("jitter"))
    shimmer = _finite_float(features.get("shimmer"))
    pitch = _finite_float(features.get("pitch_mean"))

    return VoiceAnalysis(
        mfccPattern="Stable" if score < 35 else "Variable" if score < 65 else "Abnormal",
        pitchVariation="Normal" if 70 <= pitch <= 350 else "Elevated",
        jitter="Low" if jitter < 0.03 else "Moderate" if jitter < 0.08 else "High",
        shimmer="Low" if shimmer < 0.08 else "Moderate" if shimmer < 0.18 else "High",
    )


def _probability_for_abnormal_class(model: Any, features_df: pd.DataFrame) -> float:
    probabilities = model.predict_proba(features_df)[0]
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return float(probabilities[classes.index(1)])
    if "cancer" in classes:
        return float(probabilities[classes.index("cancer")])
    if "abnormal" in classes:
        return float(probabilities[classes.index("abnormal")])
    return float(probabilities[-1])


def _extract_from_path(path: Path) -> dict[str, Any]:
    extractor = _load_voice_feature_module()
    row = pd.Series(
        {
            "file_path": str(path),
            "relative_path": path.name,
            "split": "uploaded",
            "speaker_id": "uploaded",
            "gender": "unknown",
            "voice_label": "unknown",
        }
    )
    return extractor.extract_features(row)


def predict_voice_file_path(path: Path) -> VoicePredictionResponse:
    try:
        features = _extract_from_path(path)
        payload = _load_voice_model()
        feature_columns = list(payload["feature_columns"])
        model = payload["model"]
        row = {column: _finite_float(features.get(column)) for column in feature_columns}
        features_df = pd.DataFrame([row], columns=feature_columns)
        score = round(_probability_for_abnormal_class(model, features_df) * 100, 2)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unable to process uploaded voice file: {exc}") from exc

    raw_features = {
        "pitch_mean": round(_finite_float(features.get("pitch_mean")), 2),
        "jitter": round(_finite_float(features.get("jitter")), 4),
        "shimmer": round(_finite_float(features.get("shimmer")), 4),
        "spectral_centroid_mean": round(_finite_float(features.get("spectral_centroid_mean")), 2),
    }

    return VoicePredictionResponse(
        voiceScore=score,
        voiceLabel=_voice_label(score),
        voiceAnalysis=_voice_analysis(score, features),
        rawFeatures=raw_features,
        disclaimer=settings.disclaimer,
    )


async def predict_uploaded_voice(file: UploadFile) -> VoicePredictionResponse:
    filename = file.filename or "upload.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .wav audio uploads are supported.")

    suffix = Path(filename).suffix or ".wav"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)
        return predict_voice_file_path(temp_path)
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
