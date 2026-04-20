from functools import lru_cache
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    def __init__(self) -> None:
        self.model_path = Path(
            os.getenv(
                "MODEL_PATH",
                str(PROJECT_ROOT / "model" / "roberta_ai_detector_final"),
            )
        )
        self.api_title = "RoBERTa AI Text Detector"
        self.default_chunk_tokens = int(os.getenv("CHUNK_TOKENS", "512"))
        self.default_overlap_tokens = int(os.getenv("OVERLAP_TOKENS", "64"))
        self.default_group_size = int(os.getenv("GROUP_SIZE", "3"))
        self.suspicious_ai_threshold = float(os.getenv("SUSPICIOUS_AI_THRESHOLD", "85"))
        self.suspicious_delta_threshold = float(os.getenv("SUSPICIOUS_DELTA_THRESHOLD", "20"))
        self.max_characters = int(os.getenv("MAX_CHARACTERS", "120000"))
        self.use_quantized = os.getenv("USE_QUANTIZED", "true").lower() in {"1", "true", "yes"}
        self.allowed_origins = [
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
