# Project Details: RoBERTa AI Text Detector

## 1. Project Overview

This project is a full-stack AI-text detection application built around a locally fine-tuned RoBERTa model. The system accepts pasted text or a `.txt` file, analyzes it with a transformer sequence-classification model, and returns:

- an overall prediction: `human`, `ai`, or `uncertain`
- AI probability and human probability
- confidence score
- number of text chunks analyzed
- per-chunk AI/human probabilities
- grouped suspicious regions where nearby chunks have unusually high AI probability compared with the whole document

The final product is split into two deployable pieces:

- Backend: FastAPI service that loads the model and runs inference.
- Frontend: Vite React interface that calls the backend and visualizes the result.

The intended deployment path is:

- Backend on Hugging Face Spaces using Docker.
- Frontend on Vercel.
- Large model file tracked with Git LFS or hosted through Hugging Face model/Space storage.

## 2. Current Folder Structure

```text
PROJECTSEM7/
  backend/
    app/
      __init__.py
      chunking.py
      config.py
      main.py
      model_service.py
      schemas.py
    tests/
      test_chunking.py
    .env.example
    Dockerfile
    requirements.txt

  frontend/
    src/
      api.ts
      App.tsx
      main.tsx
      styles.css
      types.ts
      vite-env.d.ts
    .env.example
    index.html
    package.json
    package-lock.json
    tsconfig.json
    vite.config.ts

  model/
    roberta_ai_detector_final/
      config.json
      merges.txt
      model.safetensors
      special_tokens_map.json
      tokenizer.json
      tokenizer_config.json
      vocab.json

  notebooks/
    training_roberta.ipynb
    inference_demo.ipynb
    external_model_experiment.ipynb
    hybrid_roberta_tfidf_experiment.ipynb

  samples/
    constitution.txt

  docs/
    deployment.md
    model-card.md
    superpowers/plans/2026-04-20-roberta-detector-fullstack.md

  .gitattributes
  .gitignore
  Dockerfile
  README.md
  project_details.md
```

## 3. What Was Cleaned Up

The original folder contained training notebooks, a final model folder, and several intermediate training checkpoints. The largest unnecessary files were:

- `optimizer.pt`
- `scheduler.pt`
- `rng_state.pth`
- `scaler.pt`
- repeated checkpoint `model.safetensors` files

Those files are useful only for resuming training. They are not needed for inference or deployment. After cleanup, only the final Hugging Face-compatible model folder remains in `model/roberta_ai_detector_final`.

The project size dropped from roughly 7.44 GB to roughly 0.47 GB after removing intermediate checkpoints.

## 4. Model Summary

The deployed model is a binary RoBERTa sequence classifier.

Important model facts:

- Architecture: `RobertaForSequenceClassification`
- Base model: `roberta-base`
- Task: text classification
- Number of classes: 2
- Label `0`: human-written text
- Label `1`: AI-generated text
- Model file: `model/roberta_ai_detector_final/model.safetensors`
- Tokenizer max length: 512 model tokens
- Saved format: Hugging Face Transformers format

The model can be loaded with:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("model/roberta_ai_detector_final")
model = AutoModelForSequenceClassification.from_pretrained("model/roberta_ai_detector_final")
```

## 5. Dataset Used

The main training notebook uses this Hugging Face dataset:

```text
shahxeebhassan/human_vs_ai_sentences
```

The notebook loads the dataset with:

```python
from datasets import load_dataset

dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences", split="train")
df = dataset.to_pandas()
```

The notebook samples 10,000 rows:

```python
df_sampled = df.sample(n=10000, random_state=42).reset_index(drop=True)
```

Then it creates an 80/20 split:

```python
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sampled["text"].tolist(),
    df_sampled["label"].tolist(),
    test_size=0.2,
    random_state=42,
)
```

This produces:

- 8,000 training samples
- 2,000 testing samples

## 6. Training Method

The notebook first trains a baseline model:

- Vectorizer: `TfidfVectorizer`
- Classifier: `LogisticRegression`
- Purpose: compare a simple statistical baseline against the transformer model

Then it fine-tunes RoBERTa:

```python
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

Training uses Hugging Face `Trainer`:

```python
training_args = TrainingArguments(
    output_dir="./roberta_ai_detector",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,
)
```

The final saved checkpoint metadata shows a later run reaching 3 epochs and 4,500 global steps.

## 7. Which Layers Were Trained

The notebook creates the model with:

```python
AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
```

No code freezes any RoBERTa layers. That means the full model was fine-tuned:

- token embeddings
- positional embeddings
- all 12 RoBERTa transformer encoder layers
- self-attention projections in each layer
- feed-forward/intermediate layers in each layer
- layer normalization parameters
- final sequence-classification head

The classification head is newly initialized when loading `roberta-base` because the base model was not originally trained for this exact binary classification task. Transformers prints a warning like this during training:

```text
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized:
classifier.dense.bias
classifier.dense.weight
classifier.out_proj.bias
classifier.out_proj.weight
```

That means the original RoBERTa body starts from pretrained language understanding weights, while the classification head starts randomly and learns the human-vs-AI decision boundary during fine-tuning.

Unless layers are explicitly frozen, Hugging Face `Trainer` updates every trainable parameter during backpropagation. In this project, no layer freezing was used, so the final model is a fully fine-tuned RoBERTa classifier, not just a trained head on top of frozen embeddings.

## 8. Training Metrics

From the notebook outputs:

Baseline TF-IDF logistic regression:

- Accuracy: 82.65%
- Human precision: 0.83
- Human recall: 0.81
- AI precision: 0.83
- AI recall: 0.84

Fine-tuned RoBERTa:

- Accuracy: 91.30%
- Human precision: 0.97
- Human recall: 0.85
- AI precision: 0.87
- AI recall: 0.97

The RoBERTa model performs much better than the TF-IDF baseline, especially on AI-text recall. High AI recall means it catches most AI examples in the test split, but the lower AI precision means some human examples may be falsely flagged as AI.

This is why the frontend and docs describe the output as probabilistic rather than absolute proof.

## 9. Why Long-Text Chunking Is Needed

RoBERTa has a fixed context window. The tokenizer config reports:

```json
"model_max_length": 512
```

A naive inference function would tokenize a long document and truncate it to the first 512 tokens. That is bad for document analysis because:

- the model ignores most of the document
- suspicious sections later in the file are missed
- results are biased toward the beginning of the text
- large uploads such as `samples/constitution.txt` are not actually analyzed as a full document

The backend solves this by splitting raw token IDs into overlapping windows.

The chunking code lives in:

```text
backend/app/chunking.py
```

The key function is:

```python
build_token_windows(token_ids, max_model_tokens=512, overlap_tokens=64)
```

RoBERTa needs special tokens around the text, so a 512-token model window can carry 510 raw text tokens. The function reserves room for those special tokens.

With:

- model window: 512
- usable payload: 510
- overlap: 64

The effective step size is:

```text
510 - 64 = 446 raw tokens
```

So the windows look like:

```text
Window 0: tokens 0-510
Window 1: tokens 446-956
Window 2: tokens 892-1402
...
```

Overlap helps avoid boundary problems where one important phrase is split across two chunks.

## 10. How Document-Level Scoring Works

Each chunk is scored separately by the model.

For each chunk, the model returns logits:

```text
logit[0] = human class score
logit[1] = AI class score
```

The backend applies softmax:

```python
probabilities = F.softmax(outputs.logits, dim=1)
```

That produces:

```text
human_probability
ai_probability
```

The document score is a weighted average of chunk scores. Chunks are weighted by token count, so a tiny final chunk does not count as much as a full-size chunk.

Example:

```text
Chunk 1: 510 tokens, 80% AI
Chunk 2: 510 tokens, 70% AI
Chunk 3: 100 tokens, 10% AI
```

The document score is not a plain average of 80, 70, and 10. It weights by token count:

```text
weighted_ai = ((510 * 80) + (510 * 70) + (100 * 10)) / (510 + 510 + 100)
```

This produces a more faithful document-level probability.

## 11. Prediction Labels

The backend converts probabilities into one of three labels:

- `ai`
- `human`
- `uncertain`

The logic lives in:

```text
backend/app/chunking.py
```

The function is:

```python
prediction_from_probabilities(ai_probability, human_probability)
```

Current rule:

- If max confidence is below 60%, return `uncertain`.
- If the difference between AI and human probability is below 10 points, return `uncertain`.
- Otherwise return whichever class has the higher probability.

This prevents the app from pretending that a 52/48 split is a confident result.

## 12. Suspicious Region Detection

The project does more than return one document-level number. It also finds grouped chunks that look unusually AI-heavy.

This matters because a document can be mixed:

- mostly human-written
- with one inserted AI-generated section
- or mostly AI-generated with human edits

The suspicious-region logic is in:

```text
backend/app/chunking.py
```

The key function is:

```python
find_suspicious_regions(
    chunks,
    document_ai_probability,
    group_size=3,
    ai_threshold=85.0,
    delta_threshold=20.0,
)
```

Default behavior:

- group 3 adjacent chunks
- calculate the group's weighted average AI probability
- flag the group if it is at least 85% AI
- also require it to be at least 20 percentage points higher than the document average

This avoids flagging high-AI sections when the whole document is already high-AI. The goal is to detect unusually suspicious regions, not just repeat the document-level verdict.

Example:

```text
Document average: 55% AI
Chunks 5-7 average: 91% AI
Delta: +36 percentage points
```

That region is flagged.

Example that is not flagged:

```text
Document average: 82% AI
Chunks 5-7 average: 88% AI
Delta: +6 percentage points
```

That is high AI, but it is not unusual compared with the rest of the document.

## 13. Backend Architecture

The backend is a FastAPI app.

Main file:

```text
backend/app/main.py
```

Routes:

```text
GET  /health
GET  /model-info
POST /predict
POST /predict-file
```

### `/health`

Returns:

```json
{
  "status": "ok",
  "model_path_exists": true,
  "quantization_enabled": true
}
```

This is used by the frontend to show whether the backend is reachable.

### `/model-info`

Loads the model service and returns model metadata:

- model architecture name
- model path
- quantization state
- tokenizer/model max length
- device

### `/predict`

Accepts JSON:

```json
{
  "text": "Text to analyze",
  "chunk_tokens": 512,
  "overlap_tokens": 64,
  "group_size": 3
}
```

Returns:

```json
{
  "prediction": "human",
  "ai_probability": 1.25,
  "human_probability": 98.75,
  "confidence": 98.75,
  "chunks_analyzed": 1,
  "suspicious_regions": [],
  "chunk_results": [],
  "model": {},
  "notes": []
}
```

### `/predict-file`

Accepts a UTF-8 text file upload and internally calls the same prediction path.

## 14. Backend Config

Config lives in:

```text
backend/app/config.py
```

Environment variables:

```text
MODEL_PATH
USE_QUANTIZED
CHUNK_TOKENS
OVERLAP_TOKENS
GROUP_SIZE
SUSPICIOUS_AI_THRESHOLD
SUSPICIOUS_DELTA_THRESHOLD
MAX_CHARACTERS
ALLOWED_ORIGINS
```

Defaults:

```text
MODEL_PATH=../model/roberta_ai_detector_final
USE_QUANTIZED=true
CHUNK_TOKENS=512
OVERLAP_TOKENS=64
GROUP_SIZE=3
SUSPICIOUS_AI_THRESHOLD=85
SUSPICIOUS_DELTA_THRESHOLD=20
MAX_CHARACTERS=120000
ALLOWED_ORIGINS=*
```

## 15. Model Loading

Model loading is handled in:

```text
backend/app/model_service.py
```

The service is cached in `main.py`:

```python
@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService(settings)
```

This means the model is loaded once per backend process, not once per request.

That matters because loading a 499 MB transformer model is expensive. If the backend loaded it for every request, the app would be slow and likely unusable on a free CPU container.

## 16. Device Selection

The backend chooses the best available device:

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

On local systems with a CUDA GPU, inference can run on GPU.

On Hugging Face free CPU Spaces, inference runs on CPU.

## 17. Quantization

The backend supports dynamic CPU quantization:

```python
torch.quantization.quantize_dynamic(
    self.model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

This applies quantization to `Linear` layers when the backend is running on CPU and `USE_QUANTIZED=true`.

Why it helps:

- lowers memory pressure
- can improve CPU inference performance
- makes free-tier deployment more practical

Why it is only applied on CPU:

- this dynamic quantization path is meant for CPU inference
- CUDA inference should use the original model weights

Important: quantization changes the inference representation, not the training process. The original source model remains the saved `model.safetensors` file.

## 18. Text-Only Transformers Patch

The backend includes this line:

```python
transformers_import_utils._torchvision_available = False
```

Reason:

Some local Python environments have mismatched `torch` and `torchvision` builds. Transformers may try to import image-related modules even when loading a text-only RoBERTa model. Since this app does not use vision models, the backend explicitly disables Transformers' torchvision availability flag before loading RoBERTa.

This prevents a broken local `torchvision` install from blocking text inference.

## 19. Frontend Architecture

The frontend is a Vite React app.

Important files:

```text
frontend/src/App.tsx
frontend/src/api.ts
frontend/src/types.ts
frontend/src/styles.css
```

### `App.tsx`

Contains the main interface:

- text area
- `.txt` upload control
- word and character counters
- backend readiness indicator
- analyze button
- empty state
- loading state
- result screen
- probability meters
- summary metrics
- flagged region cards
- chunk map visualization

### `api.ts`

Defines the API base URL:

```typescript
const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:7860";
```

For local development, it calls:

```text
http://127.0.0.1:7860
```

For Vercel deployment, set:

```text
VITE_API_URL=https://your-huggingface-space.hf.space
```

### `types.ts`

Defines TypeScript types matching the FastAPI JSON response:

- `PredictionLabel`
- `ChunkResult`
- `SuspiciousRegion`
- `AnalyzeResponse`

### `styles.css`

Provides the full responsive UI styling:

- two-panel desktop layout
- single-column mobile layout
- polished cards
- probability meters
- custom CSS icons for upload and analyze actions
- animated result entrance
- chunk bar visualization

## 20. Frontend User Flow

1. User opens the frontend.
2. Frontend calls `/health`.
3. If backend responds, status shows `Ready`.
4. User pastes text or uploads a `.txt` file.
5. User clicks `Analyze`.
6. Frontend sends a `POST /predict` request.
7. Backend loads or reuses the model service.
8. Backend tokenizes and chunks the document.
9. Backend runs RoBERTa inference.
10. Backend returns probabilities and chunk data.
11. Frontend displays:
    - verdict
    - AI probability ring
    - human/AI probability meters
    - chunk count
    - confidence
    - flagged region count
    - suspicious region cards
    - chunk map

## 21. Testing

Backend tests live in:

```text
backend/tests/test_chunking.py
```

They test:

- token window construction
- special-token space reservation
- overlap behavior
- invalid overlap validation
- weighted averaging by token count
- uncertain/human/AI label conversion
- suspicious-region detection
- region merging
- document-delta threshold behavior

Run:

```powershell
$env:PYTHONPATH="C:\Projects\PROJECTSEM7\backend"
python -m pytest C:\Projects\PROJECTSEM7\backend\tests -q
```

Expected result:

```text
6 passed
```

Frontend build:

```powershell
cd C:\Projects\PROJECTSEM7\frontend
npm run build
```

Expected result:

```text
vite build completes successfully
```

## 22. Local Development

### Backend

```powershell
cd C:\Projects\PROJECTSEM7\backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Backend URL:

```text
http://localhost:7860
```

Health check:

```text
http://localhost:7860/health
```

Interactive API docs:

```text
http://localhost:7860/docs
```

### Frontend

```powershell
cd C:\Projects\PROJECTSEM7\frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Frontend URL:

```text
http://localhost:5173
```

## 23. Docker

There are two Dockerfiles:

```text
Dockerfile
backend/Dockerfile
```

The root `Dockerfile` is the one intended for Hugging Face Spaces from the repository root.

It:

1. Starts from Python 3.10 slim.
2. Sets environment variables.
3. Uses `/app` as the working directory.
4. Installs backend dependencies.
5. Copies `backend/`.
6. Copies `model/`.
7. Starts Uvicorn on port 7860.

Important Docker environment values:

```dockerfile
ENV MODEL_PATH=/app/model/roberta_ai_detector_final
ENV USE_QUANTIZED=true
ENV PORT=7860
```

Startup command:

```dockerfile
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
```

This is compatible with Hugging Face Docker Spaces because Spaces expose a web app on port 7860 by default.

## 24. Hugging Face Spaces Deployment

The root `README.md` includes Hugging Face Space metadata:

```yaml
---
title: RoBERTa AI Text Detector API
colorFrom: teal
colorTo: red
sdk: docker
app_port: 7860
---
```

Hugging Face uses this metadata to understand that the repo is a Docker Space and that the app listens on port 7860.

Recommended Space setup:

1. Create a new Hugging Face Space.
2. Choose Docker as the SDK.
3. Push this repository content to the Space repository.
4. Make sure Git LFS is enabled for `model.safetensors`.
5. Keep `USE_QUANTIZED=true` for free CPU inference.
6. Open `/health` on the Space URL to confirm the backend is alive.

Example Space URL:

```text
https://your-space-name.hf.space
```

Example health URL:

```text
https://your-space-name.hf.space/health
```

## 25. Vercel Deployment

The frontend should be deployed separately to Vercel.

Vercel settings:

```text
Root directory: frontend
Build command: npm run build
Output directory: dist
```

Environment variable:

```text
VITE_API_URL=https://your-space-name.hf.space
```

The frontend does not contain the model. It only calls the backend API.

## 26. GitHub and Git LFS

The model file is about 499 MB. GitHub rejects normal Git blobs over 100 MB, so the model must be tracked using Git LFS or omitted from GitHub and hosted on Hugging Face instead.

This project includes:

```text
.gitattributes
```

with:

```text
model/roberta_ai_detector_final/model.safetensors filter=lfs diff=lfs merge=lfs -text
```

That tells Git LFS to store the large model file as an LFS object instead of a normal Git blob.

Before pushing:

```powershell
git lfs install
git lfs track "model/roberta_ai_detector_final/model.safetensors"
git add .gitattributes
```

Then commit normally.

## 27. Why Backend and Frontend Are Separate

The model is too large and too slow for a normal frontend-only deployment.

Vercel is excellent for static/frontend apps, but it is not ideal for serving a 499 MB PyTorch model from a serverless function because:

- cold starts would be slow
- model loading is memory-heavy
- deployment package size can be problematic
- CPU inference can exceed function expectations

Hugging Face Spaces is better for the backend because:

- it is designed for ML demos and model-serving apps
- it supports Docker
- it supports large model files through Git LFS
- free CPU hardware can run quantized inference for demos

## 28. API Response Design

The backend response is intentionally verbose enough for the UI and future extensions.

Main fields:

```text
prediction
ai_probability
human_probability
confidence
chunks_analyzed
suspicious_regions
chunk_results
model
notes
```

The frontend currently displays:

- `prediction`
- `ai_probability`
- `human_probability`
- `confidence`
- `chunks_analyzed`
- `suspicious_regions`
- `chunk_results`
- `notes`, only when present

## 29. Known Limitations

The model has real limitations:

- It was trained on sentence-style examples, not every writing domain.
- It may misclassify formal legal or academic writing as AI-like.
- It may miss heavily edited AI output.
- It may produce unstable results on very short input.
- It should not be used as the only evidence for misconduct or authorship claims.
- Quantized inference can slightly change probabilities compared with full precision.

The frontend and model card preserve this framing.

## 30. Future Improvements

Possible future upgrades:

- Add PDF/DOCX extraction.
- Add paragraph-level text highlighting.
- Add downloadable JSON/PDF reports.
- Add confidence calibration using a validation set.
- Add a second model for ensemble scoring.
- Add ONNX export for faster CPU inference.
- Add a saved benchmark comparing original vs quantized scores.
- Add authentication/rate limiting for public deployment.
- Add Redis caching for repeated documents.

The current scope intentionally uses only the fine-tuned RoBERTa model to keep deployment realistic and understandable.
