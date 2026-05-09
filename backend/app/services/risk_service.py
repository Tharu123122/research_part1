import importlib.util
from functools import lru_cache
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.prediction import PredictionResponse, RiskFactorsRequest, RiskFactorSummary
from app.utils.paths import PREVENTIVE_ENGINE_PATH


YES_VALUES = {"yes", "y", "true", "1", "current", "daily", "frequent", "regular", "occasional", "mild", "moderate", "severe", "poor"}
NO_VALUES = {"no", "n", "false", "0", "never", "none", "normal", "good", "low-risk"}

ALLOWED_VALUES = {
    "gender": {"male", "female", "other", "prefer not to say"},
    "smoking": YES_VALUES | NO_VALUES | {"former"},
    "alcohol": YES_VALUES | NO_VALUES | {"rare", "social"},
    "betelChewing": YES_VALUES | NO_VALUES | {"former"},
    "oralUlcer": YES_VALUES | NO_VALUES,
    "gumDisease": YES_VALUES | NO_VALUES,
    "oralPain": YES_VALUES | NO_VALUES,
    "hpvInfection": YES_VALUES | NO_VALUES | {"unknown"},
    "poorOralHygiene": YES_VALUES | NO_VALUES,
    "diet": {"low", "medium", "high", "poor", "average", "good"},
    "familyHistory": YES_VALUES | NO_VALUES | {"unknown"},
    "compromisedImmuneSystem": YES_VALUES | NO_VALUES | {"unknown"},
    "unexplainedBleeding": YES_VALUES | NO_VALUES,
    "difficultySwallowing": YES_VALUES | NO_VALUES,
    "whiteOrRedPatches": YES_VALUES | NO_VALUES,
}


def _clean(value: str) -> str:
    return str(value).strip()


def _key(value: str) -> str:
    return _clean(value).lower()


def _validate_categories(payload: RiskFactorsRequest) -> None:
    errors: list[str] = []
    for field_name, allowed in ALLOWED_VALUES.items():
        value = _key(getattr(payload, field_name))
        if value not in allowed:
            errors.append(f"{field_name} has unsupported value '{getattr(payload, field_name)}'")
    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"errors": errors})


def _truthy_for_engine(value: str) -> str:
    value_key = _key(value)
    if value_key in YES_VALUES or value_key == "former":
        return "Yes"
    return "No"


def _diet_for_engine(value: str) -> str:
    value_key = _key(value)
    if value_key in {"low", "poor"}:
        return "Low"
    if value_key in {"high", "good"}:
        return "High"
    return "Medium"


def _hygiene_signal(gum_disease: str, poor_oral_hygiene: str) -> str:
    return "Yes" if _truthy_for_engine(gum_disease) == "Yes" or _truthy_for_engine(poor_oral_hygiene) == "Yes" else "No"


@lru_cache(maxsize=1)
def _load_preventive_engine() -> Any:
    if not PREVENTIVE_ENGINE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preventive risk engine file is missing.",
        )
    spec = importlib.util.spec_from_file_location("preventive_risk_engine", PREVENTIVE_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Preventive risk engine cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "calculate_preventive_risk"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Preventive risk function is unavailable.")
    return module


def map_to_engine_fields(payload: RiskFactorsRequest) -> dict[str, Any]:
    return {
        "Age": payload.age,
        "Gender": payload.gender,
        "Tobacco Use": _truthy_for_engine(payload.smoking),
        "Alcohol Consumption": _truthy_for_engine(payload.alcohol),
        "Betel Quid Use": _truthy_for_engine(payload.betelChewing),
        "Oral Lesions": _truthy_for_engine(payload.oralUlcer),
        "Poor Oral Hygiene": _hygiene_signal(payload.gumDisease, payload.poorOralHygiene),
        "HPV Infection": _truthy_for_engine(payload.hpvInfection),
        "Diet (Fruits & Vegetables Intake)": _diet_for_engine(payload.diet),
        "Family History of Cancer": _truthy_for_engine(payload.familyHistory),
        "Compromised Immune System": _truthy_for_engine(payload.compromisedImmuneSystem),
        "Unexplained Bleeding": _truthy_for_engine(payload.unexplainedBleeding),
        "Difficulty Swallowing": _truthy_for_engine(payload.difficultySwallowing),
        "White or Red Patches in Mouth": _truthy_for_engine(payload.whiteOrRedPatches),
    }


def level_from_score(score: float) -> str:
    if score < 35:
        return "low"
    if score < 65:
        return "moderate"
    return "high"


def risk_summary(payload: RiskFactorsRequest) -> RiskFactorSummary:
    return RiskFactorSummary(
        age=payload.age,
        smoking=payload.smoking,
        alcohol=payload.alcohol,
        betelChewing=payload.betelChewing,
        oralUlcer=payload.oralUlcer,
        gumDisease=payload.gumDisease,
        oralPain=payload.oralPain,
    )


def _symptom_count(payload: RiskFactorsRequest) -> int:
    symptom_values = [
        payload.oralUlcer,
        payload.oralPain,
        payload.unexplainedBleeding,
        payload.difficultySwallowing,
        payload.whiteOrRedPatches,
    ]
    return sum(_truthy_for_engine(value) == "Yes" for value in symptom_values)


def build_structured_prediction(payload: RiskFactorsRequest) -> PredictionResponse:
    _validate_categories(payload)
    engine = _load_preventive_engine()
    result = engine.calculate_preventive_risk(map_to_engine_fields(payload))
    structured_score = float(result.score)
    insights = list(result.reasons)

    if _truthy_for_engine(payload.oralPain) == "Yes":
        insights.append("Oral pain was reported and should be monitored if persistent.")
    if _symptom_count(payload) < 2:
        insights.append("No severe symptom cluster detected from submitted risk factors.")

    return PredictionResponse(
        riskPercentage=round(structured_score, 2),
        level=level_from_score(structured_score),
        structuredScore=round(structured_score, 2),
        voiceScore=None,
        finalScore=round(structured_score, 2),
        insights=list(dict.fromkeys(insights)),
        recommendations=list(result.recommendations),
        riskFactorSummary=risk_summary(payload),
        voiceAnalysis=None,
        disclaimer=settings.disclaimer,
    )


def combine_predictions(
    *,
    structured: PredictionResponse,
    voice_score: float | None,
    voice_analysis: Any | None,
) -> PredictionResponse:
    if voice_score is None:
        final_score = structured.structuredScore
        voice_insights = ["Voice analysis was not provided; final score uses structured risk factors only."]
    else:
        final_score = (0.70 * structured.structuredScore) + (0.30 * voice_score)
        if voice_score < 35:
            voice_insights = ["Voice signal markers are within expected range."]
        elif voice_score < 65:
            voice_insights = ["Voice signal markers show slight variation."]
        else:
            voice_insights = ["Voice signal markers include abnormality indicators."]

    recommendations = list(structured.recommendations)
    if structured.level in {"moderate", "high"} or (voice_score is not None and voice_score >= 35):
        recommendations.append("Consult a qualified healthcare professional for persistent symptoms.")

    final_score = round(final_score, 2)
    return structured.model_copy(
        update={
            "riskPercentage": final_score,
            "level": level_from_score(final_score),
            "voiceScore": round(float(voice_score), 2) if voice_score is not None else None,
            "finalScore": final_score,
            "insights": list(dict.fromkeys(voice_insights + structured.insights)),
            "recommendations": list(dict.fromkeys(recommendations)),
            "voiceAnalysis": voice_analysis,
        }
    )
