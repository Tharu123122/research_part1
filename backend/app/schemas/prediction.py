from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


RiskLevel = Literal["low", "moderate", "high"]
VoiceLabel = Literal["stable", "slight_variation", "abnormal_marker"]


class RiskFactorsRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: str
    smoking: str
    alcohol: str
    betelChewing: str
    oralUlcer: str
    gumDisease: str
    oralPain: str
    hpvInfection: str
    poorOralHygiene: str
    diet: str
    familyHistory: str
    compromisedImmuneSystem: str
    unexplainedBleeding: str
    difficultySwallowing: str
    whiteOrRedPatches: str

    @field_validator("*", mode="before")
    @classmethod
    def strip_string_values(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class RiskFactorSummary(BaseModel):
    age: int
    smoking: str
    alcohol: str
    betelChewing: str
    oralUlcer: str
    gumDisease: str
    oralPain: str


class VoiceAnalysis(BaseModel):
    mfccPattern: str
    pitchVariation: str
    jitter: str
    shimmer: str


class PredictionResponse(BaseModel):
    riskPercentage: float
    level: RiskLevel
    structuredScore: float
    voiceScore: float | None
    finalScore: float
    insights: list[str]
    recommendations: list[str]
    riskFactorSummary: RiskFactorSummary
    voiceAnalysis: VoiceAnalysis | None
    disclaimer: str


class VoicePredictionResponse(BaseModel):
    voiceScore: float
    voiceLabel: VoiceLabel
    voiceAnalysis: VoiceAnalysis
    rawFeatures: dict[str, float]
    disclaimer: str


class MultimodalJsonRequest(BaseModel):
    riskFactors: RiskFactorsRequest
    voiceScore: float | None = Field(default=None, ge=0, le=100)


class HistoryItem(BaseModel):
    id: str
    date: str
    riskPercentage: float
    level: RiskLevel
    summary: str
    voiceStatus: str


class PredictionRecord(BaseModel):
    id: str
    date: str
    predictionType: Literal["risk_factors", "voice", "multimodal"]
    request: dict[str, Any]
    response: dict[str, Any]
    history: HistoryItem | None = None
