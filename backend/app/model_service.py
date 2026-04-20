from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .chunking import (
    ScoreLike,
    build_token_windows,
    find_suspicious_regions,
    prediction_from_probabilities,
    weighted_average,
)
from .config import Settings


# This is a text-only service. Some local Windows Python environments have a
# mismatched torchvision build, and Transformers can otherwise import it while
# resolving RoBERTa even though no vision feature is needed.
transformers_import_utils._torchvision_available = False


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: str
    quantized: bool
    max_model_tokens: int
    device: str


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantized = False
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)

        if settings.use_quantized and self.device.type == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            self.quantized = True

        self.model.to(self.device)
        self.model.eval()

    def analyze(
        self,
        text: str,
        chunk_tokens: int,
        overlap_tokens: int,
        group_size: int,
    ) -> dict:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        windows = build_token_windows(token_ids, chunk_tokens, overlap_tokens)
        if not windows:
            raise ValueError("Text did not produce analyzable tokens.")

        input_ids = [
            self.tokenizer.build_inputs_with_special_tokens(window.token_ids)
            for window in windows
        ]
        attention_masks = [[1] * len(ids) for ids in input_ids]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.no_grad():
            logits = self.model(**batch).logits
            probabilities = F.softmax(logits, dim=1).cpu().tolist()

        chunk_results: list[ScoreLike] = []
        for window, probability in zip(windows, probabilities):
            human_probability = round(probability[0] * 100, 2)
            ai_probability = round(probability[1] * 100, 2)
            chunk_results.append(
                ScoreLike(
                    index=window.index,
                    start_token=window.start_token,
                    end_token=window.end_token,
                    token_count=window.token_count,
                    ai_probability=ai_probability,
                    human_probability=human_probability,
                )
            )

        document_ai = weighted_average(chunk_results, "ai_probability")
        document_human = round(100 - document_ai, 2)
        prediction = prediction_from_probabilities(document_ai, document_human)
        confidence = round(max(document_ai, document_human), 2)
        suspicious_regions = find_suspicious_regions(
            chunk_results,
            document_ai_probability=document_ai,
            group_size=group_size,
            ai_threshold=self.settings.suspicious_ai_threshold,
            delta_threshold=self.settings.suspicious_delta_threshold,
        )

        notes = [
            "AI-text detection is probabilistic and should not be used as the only evidence for authorship decisions.",
        ]
        if len(windows) == 1 and len(token_ids) < 80:
            notes.append("Very short text can produce unstable detector scores.")

        return {
            "prediction": prediction,
            "ai_probability": document_ai,
            "human_probability": document_human,
            "confidence": confidence,
            "chunks_analyzed": len(chunk_results),
            "suspicious_regions": suspicious_regions,
            "chunk_results": [
                {
                    "index": chunk.index,
                    "start_token": chunk.start_token,
                    "end_token": chunk.end_token,
                    "token_count": chunk.token_count,
                    "ai_probability": chunk.ai_probability,
                    "human_probability": chunk.human_probability,
                    "prediction": prediction_from_probabilities(
                        chunk.ai_probability,
                        chunk.human_probability,
                    ),
                }
                for chunk in chunk_results
            ],
            "model": self.info().__dict__,
            "notes": notes,
        }

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name(),
            path=str(self.settings.model_path),
            quantized=self.quantized,
            max_model_tokens=int(getattr(self.tokenizer, "model_max_length", 512)),
            device=str(self.device),
        )

    def _model_name(self) -> str:
        config_path = Path(self.settings.model_path) / "config.json"
        if not config_path.exists():
            return "Fine-tuned RoBERTa detector"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            architecture = config.get("architectures", ["Fine-tuned RoBERTa detector"])[0]
            return str(architecture)
        except json.JSONDecodeError:
            return "Fine-tuned RoBERTa detector"
