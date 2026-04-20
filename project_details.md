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

Current live deployment:

- Public app: `https://frontend-navy-seven-60.vercel.app`
- Backend API: `https://clashking-aidetect.hf.space`
- Backend health check: `https://clashking-aidetect.hf.space/health`
- Hugging Face Space: `https://huggingface.co/spaces/clashking/AIDetect`

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
const API_URL =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.PROD ? "https://clashking-aidetect.hf.space" : "http://127.0.0.1:7860");
```

For local development, it calls:

```text
http://127.0.0.1:7860
```

For Vercel production, it falls back to:

```text
https://clashking-aidetect.hf.space
```

`VITE_API_URL` can still be set if a future deployment needs to point the frontend at a
different backend.

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

The backend is deployed as a public Hugging Face Docker Space:

```text
Space repository: https://huggingface.co/spaces/clashking/AIDetect
Live API: https://clashking-aidetect.hf.space
Health check: https://clashking-aidetect.hf.space/health
```

The root `README.md` contains the Space metadata that Hugging Face reads when building
the app:

```yaml
---
title: RoBERTa AI Text Detector API
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
---
```

Hugging Face uses that metadata to identify the repository as a Docker Space and route
traffic to port `7860`, which is the port exposed by the FastAPI container.

The deployment used the locally authenticated Hugging Face CLI and Python client. The
token was never written into the project files, committed, printed into documentation, or
added to the public repository.

The Space was created with:

```powershell
hf repo create clashking/AIDetect --repo-type space --space_sdk docker --exist-ok
```

The first attempt to push the Space through Git ran into the existing starter commit in
the Hugging Face Space repository, and a larger Git push timed out. To avoid fighting the
Space Git history and to keep the deployment moving, the project was uploaded with
`huggingface_hub.HfApi.upload_folder` instead:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id="clashking/AIDetect",
    repo_type="space",
    folder_path=".",
    commit_message="Deploy RoBERTa detector Space",
    ignore_patterns=[
        ".git/*",
        ".pytest_cache/*",
        "frontend/node_modules/*",
        "frontend/dist/*",
        "**/__pycache__/*",
        "**/*.log",
        "notebooks/*",
        "samples/*",
        "docs/superpowers/*",
        "scripts/*",
    ],
)
```

The ignored paths are local development artifacts, generated frontend output, notebooks,
scratch samples, logs, Python caches, and local-only helper material that is not needed
for the hosted backend.

One Hugging Face metadata validation issue appeared during deployment: `colorFrom: teal`
is not a valid Space metadata color. It was changed to `colorFrom: green`, after which
the Space accepted and built the project.

The initial backend upload landed at this Space commit:

```text
https://huggingface.co/spaces/clashking/AIDetect/commit/5bc19107c1a58ea51c96656d893d447cb10f306b
```

After the live app was working, documentation updates were synced back to the Space with
`HfApi.upload_file` so the public Space repository stayed aligned with GitHub:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="project_details.md",
    path_in_repo="project_details.md",
    repo_id="clashking/AIDetect",
    repo_type="space",
    commit_message="Update project details",
)
```

The backend deployment was verified by requesting:

```text
https://clashking-aidetect.hf.space/health
```

The expected response confirms that the API process is alive, the model path exists, and
CPU dynamic quantization is enabled:

```json
{
  "status": "ok",
  "model_path_exists": true,
  "quantization_enabled": true
}
```

## 25. Vercel Deployment

The React frontend is deployed separately to Vercel:

```text
Live frontend: https://frontend-navy-seven-60.vercel.app
Vercel project: frontend
Vercel account/team: atharvas-projects-390e680b
Backend used by the live frontend: https://clashking-aidetect.hf.space
```

The first Vercel deployment attempt was run from the repository root:

```powershell
npx vercel deploy --prod --yes
```

That attempt was not kept because Vercel auto-detected both frontend and backend services
and inferred an invalid project setup for this repository. The root `vercel.json` was
cleaned up, and the frontend was given its own Vercel configuration file at
`frontend/vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "installCommand": "npm install",
  "framework": "vite"
}
```

The production deployment was then run from the frontend directory:

```powershell
cd C:\Projects\PROJECTSEM7\frontend
cmd /c npx vercel deploy --prod --yes
```

Vercel built the Vite app with `npm run build`, published the `dist` directory, and
created the production deployment. The deployment URL was then given the stable public
alias:

```text
https://frontend-navy-seven-60.vercel.app
```

The frontend is configured so production builds call the live Hugging Face API even if no
public Vercel environment variable is set. The relevant logic is in `frontend/src/api.ts`:

```ts
const API_URL =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.PROD ? "https://clashking-aidetect.hf.space" : "http://127.0.0.1:7860");
```

This gives the app three useful behaviors:

1. Local development can still use `http://127.0.0.1:7860`.
2. Vercel production uses `https://clashking-aidetect.hf.space` by default.
3. A future deployment can override the backend with `VITE_API_URL` without editing code.

The live frontend was verified by opening the Vercel URL, submitting text for analysis,
and confirming that the UI received a model verdict from the Hugging Face backend.

No model weights are included in the Vercel deployment. The browser only downloads the
static React app and calls the backend API.

Deployment safety checks:

1. `.vercel/` is ignored so local Vercel project metadata is not committed.
2. `.env` and `.env.local` are ignored so local credentials are not committed.
3. Hugging Face authentication is handled by the local CLI/client session, not by source files.
4. Secret scans were run before publishing to check for common token patterns such as
   `hf_`, `github_pat_`, `ghp_`, `sk-`, `TOKEN=`, `PASSWORD=`, and `SECRET=`.

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

## 28. Why This Stack Was Chosen

This project was designed as a deployable ML product, not only as a training notebook. The
stack choices are intentionally practical: they keep the system lightweight enough for free
hosting while still looking and behaving like a real application.

### FastAPI instead of Django

FastAPI was chosen because this backend is an inference API, not a traditional content
management or database-heavy web application.

FastAPI fits this use case well because:

- it has first-class request and response validation through Pydantic
- it automatically generates OpenAPI documentation
- it is lightweight and starts quickly
- it works naturally with typed Python code
- it is easy to containerize for a single-purpose ML service
- it avoids bringing in an ORM, admin panel, templating system, and large project structure
  that this app does not need

Django is excellent when the application needs built-in authentication, relational database
models, an admin dashboard, server-rendered pages, permissions, migrations, and a larger
monolithic product structure. This project does not need those features in the inference
layer. Using Django here would make the service heavier without improving model serving.

Flask would also work, but FastAPI gives stronger typed schemas and better generated API
docs out of the box. For an ML API that accepts structured inputs and returns structured
model results, those built-in contracts are useful.

### Vite React instead of a full server-rendered framework

The frontend is a static client application. It does not need server-side rendering,
database access, user accounts, or private backend logic. Vite React was chosen because it
is fast, simple, and produces a clean static build that Vercel can host easily.

Next.js would be a good choice if the project needed server-rendered routes, authenticated
dashboards, API routes colocated with pages, SEO-heavy content, or database-backed user
flows. This project only needs a smooth browser interface that calls a separate ML backend,
so Vite keeps the frontend smaller and easier to reason about.

Plain HTML and JavaScript would have been enough for a basic demo, but React and TypeScript
make the UI easier to grow. The response includes nested data such as chunk results and
suspicious regions, so typed frontend models reduce mistakes when rendering the analysis.

### Hugging Face Spaces instead of generic app hosting for the backend

The backend serves a large transformer model, so the hosting target needs to tolerate model
artifacts, Python ML dependencies, a container build, and CPU inference.

Hugging Face Spaces was chosen because:

- it is built around ML demos and model-serving apps
- it supports Docker Spaces
- it handles large model files better than typical serverless frontend hosts
- it provides a public URL that can be used directly by the frontend
- it is a credible platform for showcasing machine learning projects
- the free CPU tier is enough for a demo when dynamic quantization is enabled

Render, Railway, and Fly.io are all valid alternatives for container hosting. They may be
better for always-on production services, custom domains, persistent databases, background
workers, or paid reliability. For this project, Hugging Face was the better fit because the
core asset is a transformer model and the project benefits from living in the same ecosystem
as the model tooling.

### Vercel instead of serving the frontend from FastAPI

The frontend and backend were deployed separately because each side has different runtime
needs.

Vercel is ideal for the React frontend because:

- it serves static frontend assets quickly
- it builds Vite projects cleanly
- it gives a stable public URL with almost no infrastructure work
- it avoids making the ML container also handle static asset hosting

FastAPI could serve the frontend files, but that would couple the UI release cycle to the
model-serving container. Keeping them separate makes the architecture cleaner: Vercel owns
the browser experience, and Hugging Face owns the ML inference runtime.

### Docker instead of a manual server setup

Docker was chosen because the backend depends on a specific Python ML environment, model
files, and system-level runtime behavior. A Dockerfile makes that environment repeatable.

This matters because ML projects often fail during deployment for reasons unrelated to the
model itself: dependency versions, file paths, ports, missing runtime packages, or mismatched
local and cloud environments. Docker makes the Space build closer to the local backend
setup and documents exactly how the service starts.

### Fine-tuned RoBERTa instead of calling an external LLM API

The project uses a local fine-tuned classifier rather than an external LLM API because the
task is classification, not generation.

That choice has several advantages:

- inference does not depend on a paid third-party generation API
- the model can run inside the deployed backend
- results are more reproducible than prompt-based classification
- the architecture demonstrates actual model fine-tuning and packaging
- the deployment can be shown publicly without exposing an API key

An LLM API could be useful for explanations, rubric-style review, or a second opinion, but
it would add cost, latency, and secret-management requirements. For this portfolio project,
the stronger signal is showing that a model was trained, packaged, optimized, deployed, and
served through a real API.

### Dynamic quantization instead of full precision inference

The public backend runs on CPU. Dynamic quantization was added because it can reduce memory
pressure and improve CPU inference practicality without requiring GPU hardware.

Full precision inference is useful during evaluation and when maximum numerical fidelity is
required. Quantized inference is a better fit for a free-tier public demo where the priority
is keeping the service responsive and lightweight enough to run in a constrained container.

### Chunking instead of truncating long documents

RoBERTa has a fixed token limit, so long documents cannot be passed through the classifier
as one input. The simplest option would be truncation, but truncation discards most of a
long document and can hide the parts that matter.

This project uses overlapping token windows instead. That makes the analysis more useful:
the backend scores the full document in segments, aggregates the probabilities, and reports
groups of neighboring chunks with unusually high AI probability. This turns the app from a
single-score demo into a tool that can help a user inspect where the signal appears.

## 29. API Response Design

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

## 30. Known Limitations

The model has real limitations:

- It was trained on sentence-style examples, not every writing domain.
- It may misclassify formal legal or academic writing as AI-like.
- It may miss heavily edited AI output.
- It may produce unstable results on very short input.
- It should not be used as the only evidence for misconduct or authorship claims.
- Quantized inference can slightly change probabilities compared with full precision.

The frontend and model card preserve this framing.

## 31. Future Improvements

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
