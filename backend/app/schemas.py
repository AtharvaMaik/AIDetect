from typing import Literal

from pydantic import BaseModel, Field


PredictionLabel = Literal["human", "ai", "uncertain"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    chunk_tokens: int | None = Field(default=None, ge=64, le=512)
    overlap_tokens: int | None = Field(default=None, ge=0, le=256)
    group_size: int | None = Field(default=None, ge=1, le=8)


class ChunkResult(BaseModel):
    index: int
    start_token: int
    end_token: int
    token_count: int
    ai_probability: float
    human_probability: float
    prediction: PredictionLabel


class SuspiciousRegion(BaseModel):
    start_chunk: int
    end_chunk: int
    start_token: int
    end_token: int
    ai_probability: float
    human_probability: float
    delta_from_document: float
    reason: str


class AnalyzeResponse(BaseModel):
    prediction: PredictionLabel
    ai_probability: float
    human_probability: float
    confidence: float
    chunks_analyzed: int
    suspicious_regions: list[SuspiciousRegion]
    chunk_results: list[ChunkResult]
    model: dict[str, str | int | bool | float]
    notes: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_path_exists: bool
    quantization_enabled: bool
