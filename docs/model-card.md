# RoBERTa AI Text Detector Model Card

## Model

- Architecture: `RobertaForSequenceClassification`
- Source artifact: `model/roberta_ai_detector_final`
- Base model: `roberta-base`
- Task: binary text classification
- Labels: `0 = human`, `1 = AI`
- Default inference window: 512 model tokens with 64-token overlap

## Training Summary

The training notebook samples 10,000 examples from `shahxeebhassan/human_vs_ai_sentences`, uses an 80/20 train-test split, and fine-tunes RoBERTa for human-vs-AI sentence classification.

Observed notebook metrics:

- TF-IDF logistic regression baseline accuracy: 82.65%
- Fine-tuned RoBERTa accuracy: 91.30%
- Human class: precision 0.97, recall 0.85
- AI class: precision 0.87, recall 0.97

## Inference Behavior

Long documents are split into overlapping token windows instead of being truncated. The backend calculates a weighted document score from chunk scores, then groups adjacent chunks to identify sections with unusually high AI probability compared with the document average.

CPU dynamic quantization can be enabled with `USE_QUANTIZED=true`. Quantization reduces runtime memory pressure on free-tier CPU deployments, but the original safetensors weights remain the source model artifact.

## Limitations

- AI detection is probabilistic and should not be treated as proof of authorship.
- The model was trained on sentence-style data and may generalize unevenly to legal text, academic essays, creative writing, short messages, or heavily edited AI output.
- Very short text can produce unstable scores.
- The detector can be affected by paraphrasing, translation, prompt style, and domain mismatch.

## Intended Use

Use this project as an educational and assistive signal for reviewing text. Do not use it as the sole basis for academic misconduct, employment, legal, or disciplinary decisions.
