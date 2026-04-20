import pytest

from app.chunking import (
    ScoreLike,
    build_token_windows,
    find_suspicious_regions,
    prediction_from_probabilities,
    weighted_average,
)


def test_build_token_windows_reserves_special_tokens_and_overlaps():
    token_ids = list(range(1000))

    windows = build_token_windows(token_ids, max_model_tokens=512, overlap_tokens=64)

    assert len(windows) == 3
    assert windows[0].start_token == 0
    assert windows[0].end_token == 510
    assert windows[1].start_token == 446
    assert windows[1].end_token == 956
    assert windows[2].start_token == 892
    assert windows[2].end_token == 1000


def test_build_token_windows_rejects_overlap_that_consumes_chunk():
    with pytest.raises(ValueError, match="overlap_tokens"):
        build_token_windows(list(range(20)), max_model_tokens=128, overlap_tokens=126)


def test_weighted_average_uses_token_count():
    chunks = [
        ScoreLike(0, 0, 100, 100, 90.0, 10.0),
        ScoreLike(1, 100, 300, 200, 30.0, 70.0),
    ]

    assert weighted_average(chunks, "ai_probability") == 50.0


def test_prediction_has_uncertain_band():
    assert prediction_from_probabilities(54.0, 46.0) == "uncertain"
    assert prediction_from_probabilities(70.0, 30.0) == "ai"
    assert prediction_from_probabilities(25.0, 75.0) == "human"


def test_find_suspicious_regions_merges_adjacent_high_ai_groups():
    chunks = [
        ScoreLike(0, 0, 100, 100, 20.0, 80.0),
        ScoreLike(1, 100, 200, 100, 92.0, 8.0),
        ScoreLike(2, 200, 300, 100, 94.0, 6.0),
        ScoreLike(3, 300, 400, 100, 96.0, 4.0),
        ScoreLike(4, 400, 500, 100, 25.0, 75.0),
    ]

    regions = find_suspicious_regions(
        chunks,
        document_ai_probability=55.0,
        group_size=2,
        ai_threshold=85.0,
        delta_threshold=20.0,
    )

    assert len(regions) == 1
    assert regions[0]["start_chunk"] == 1
    assert regions[0]["end_chunk"] == 3
    assert regions[0]["ai_probability"] == 94.0
    assert regions[0]["delta_from_document"] == 39.0


def test_find_suspicious_regions_requires_document_delta():
    chunks = [
        ScoreLike(0, 0, 100, 100, 88.0, 12.0),
        ScoreLike(1, 100, 200, 100, 89.0, 11.0),
    ]

    regions = find_suspicious_regions(
        chunks,
        document_ai_probability=80.0,
        group_size=2,
        ai_threshold=85.0,
        delta_threshold=20.0,
    )

    assert regions == []
