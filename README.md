---
title: RoBERTa AI Text Detector API
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
---

# RoBERTa AI Text Detector

System.Object[]

A deployed full-stack machine learning product that detects AI-generated writing with a fine-tuned RoBERTa classifier, long-document token chunking, suspicious-region grouping, a FastAPI inference backend, and a polished React interface.

This project is intentionally more than a notebook. It takes a trained transformer model all the way to a public product: model packaging, API design, CPU quantization, Docker deployment, frontend integration, live hosting, documentation, and responsible-use framing.

## Live Demo

- Public app: [https://frontend-navy-seven-60.vercel.app](https://frontend-navy-seven-60.vercel.app)
- Backend API: [https://clashking-aidetect.hf.space](https://clashking-aidetect.hf.space)
- Backend health check: [https://clashking-aidetect.hf.space/health](https://clashking-aidetect.hf.space/health)
- Hugging Face Space: [https://huggingface.co/spaces/clashking/AIDetect](https://huggingface.co/spaces/clashking/AIDetect)

## What It Does

The app accepts pasted text or uploaded text files and returns:

- an overall Human/AI prediction
- calibrated-looking AI and human probability scores
- confidence level
- number of model windows analyzed
- per-window chunk results for long documents
- grouped suspicious regions when neighboring chunks show unusually high AI probability

Long inputs are split into overlapping RoBERTa-token windows because transformer classifiers have a fixed context limit. Instead of truncating the document, the backend scores each window and aggregates the result into a document-level verdict.

## Why This Project Matters

This project demonstrates the kind of work needed to turn an ML experiment into a usable software system:

- Fine-tuned a transformer model and saved it in Hugging Face-compatible format.
- Built a production-style FastAPI backend with typed request/response schemas.
- Added overlapping token-window inference for documents longer than RoBERTa's 512-token limit.
- Grouped high-risk neighboring chunks so users can inspect suspicious sections instead of only seeing one global score.
- Applied CPU dynamic quantization to make inference lighter for free-tier hosting.
- Shipped the backend as a Dockerized Hugging Face Space.
- Shipped the frontend as a Vite React app on Vercel.
- Kept secrets out of the public repository and documented the deployment process.

## Architecture

```text
User
  |
  v
Vercel React frontend
  |
  | POST /predict or /predict-file
  v
Hugging Face Docker Space
  |
  v
FastAPI backend
  |
  v
RoBERTa tokenizer -> overlapping chunks -> fine-tuned classifier -> grouped results
```

## Tech Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Model | Fine-tuned RoBERTa sequence classifier | Strong encoder-only baseline for text classification tasks |
| Backend | FastAPI | Lightweight, typed, async-friendly API layer without Django's full web-app overhead |
| Inference | PyTorch + Transformers | Native support for the saved model and tokenizer |
| Optimization | PyTorch dynamic quantization | Smaller, lighter CPU inference for free-tier deployment |
| Frontend | Vite + React + TypeScript | Fast development, clean static build, strong UI type safety |
| Backend hosting | Hugging Face Spaces with Docker | ML-friendly hosting that supports large model artifacts and containerized inference |
| Frontend hosting | Vercel | Excellent static frontend hosting, fast builds, simple production URLs |

## Repository Structure

```text
backend/   FastAPI inference API, model loading, chunking, tests
frontend/  Vite React interface and API client
model/     Fine-tuned RoBERTa artifact
notebooks/ Training and experiment notebooks
samples/   Demo text files
docs/      Model card and deployment notes
```

## API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Confirms the backend is alive and the model path is available |
| `GET /model-info` | Returns model/runtime metadata |
| `POST /predict` | Analyzes raw text |
| `POST /predict-file` | Analyzes uploaded text files |

## Local Backend

```powershell
cd C:\Projects\PROJECTSEM7\backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 7860
```

## Local Frontend

```powershell
cd C:\Projects\PROJECTSEM7\frontend
npm install
npm run dev
```

Set `VITE_API_URL` in `frontend/.env.local` when the backend is not running on `http://127.0.0.1:7860`.

## Inference Details

- Default chunk size: 512 model tokens
- Default overlap: 64 tokens
- Default suspicious-region group size: 3 chunks
- Optional CPU dynamic quantization: `USE_QUANTIZED=true`
- Production backend: Hugging Face CPU Space
- Production frontend: Vercel static deployment

## Documentation

- [Project details](project_details.md): full architecture, training, model, deployment, and engineering tradeoff notes
- [Deployment guide](docs/deployment.md): public deployment URLs and deployment commands
- [Model card](docs/model-card.md): model behavior, intended use, and limitations

## Responsible Use

The model returns probabilistic signals, not proof of authorship. AI-text detection can be brittle across writing domains, editing styles, and short inputs. Use the output as a review aid, especially for long documents where chunk-level patterns can be more informative than a single document-level label.

## Contributing

Contributions are welcome. You can help by reporting bugs, suggesting features, improving documentation, or opening pull requests.

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Test the project locally when possible.
5. Open a pull request with a clear summary of what changed.
