from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TokenWindow:
    index: int
    token_ids: list[int]
    start_token: int
    end_token: int

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token


@dataclass(frozen=True)
class ScoreLike:
    index: int
    start_token: int
    end_token: int
    token_count: int
    ai_probability: float
    human_probability: float


def build_token_windows(
    token_ids: list[int],
    max_model_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[TokenWindow]:
    """Split raw token ids into overlapping model-size windows.

    RoBERTa needs two special tokens, so a 512-token model window carries 510
    raw text tokens. Keeping this here avoids accidental first-512 truncation.
    """
    if max_model_tokens < 4:
        raise ValueError("max_model_tokens must leave room for special tokens")

    payload_tokens = max_model_tokens - 2
    if overlap_tokens >= payload_tokens:
        raise ValueError("overlap_tokens must be smaller than usable chunk size")

    if not token_ids:
        return []

    windows: list[TokenWindow] = []
    step = payload_tokens - overlap_tokens
    start = 0

    while start < len(token_ids):
        end = min(start + payload_tokens, len(token_ids))
        windows.append(
            TokenWindow(
                index=len(windows),
                token_ids=token_ids[start:end],
                start_token=start,
                end_token=end,
            )
        )
        if end == len(token_ids):
            break
        start += step

    return windows


def weighted_average(scores: Iterable[ScoreLike], field: str) -> float:
    score_list = list(scores)
    if not score_list:
        return 0.0

    total_weight = sum(max(score.token_count, 1) for score in score_list)
    weighted_sum = sum(getattr(score, field) * max(score.token_count, 1) for score in score_list)
    return round(weighted_sum / total_weight, 2)


def prediction_from_probabilities(ai_probability: float, human_probability: float) -> str:
    confidence = max(ai_probability, human_probability)
    if confidence < 60 or abs(ai_probability - human_probability) < 10:
        return "uncertain"
    return "ai" if ai_probability > human_probability else "human"


def find_suspicious_regions(
    chunks: list[ScoreLike],
    document_ai_probability: float,
    group_size: int = 3,
    ai_threshold: float = 85.0,
    delta_threshold: float = 20.0,
) -> list[dict[str, int | float | str]]:
    if not chunks:
        return []

    group_size = max(1, min(group_size, len(chunks)))
    flagged: list[tuple[int, int, float]] = []

    for start in range(0, len(chunks) - group_size + 1):
        group = chunks[start : start + group_size]
        group_ai = weighted_average(group, "ai_probability")
        delta = round(group_ai - document_ai_probability, 2)
        if group_ai >= ai_threshold and delta >= delta_threshold:
            flagged.append((start, start + group_size - 1, group_ai))

    if not flagged:
        return []

    merged: list[tuple[int, int]] = []
    for start, end, _score in flagged:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))

    regions: list[dict[str, int | float | str]] = []
    for start, end in merged:
        group = chunks[start : end + 1]
        ai_probability = weighted_average(group, "ai_probability")
        human_probability = round(100 - ai_probability, 2)
        delta = round(ai_probability - document_ai_probability, 2)
        regions.append(
            {
                "start_chunk": group[0].index,
                "end_chunk": group[-1].index,
                "start_token": group[0].start_token,
                "end_token": group[-1].end_token,
                "ai_probability": ai_probability,
                "human_probability": human_probability,
                "delta_from_document": delta,
                "reason": (
                    f"Grouped chunks average {ai_probability:.2f}% AI, "
                    f"{delta:.2f} points above the document average."
                ),
            }
        )

    return regions
