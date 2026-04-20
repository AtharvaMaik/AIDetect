---
title: RoBERTa AI Text Detector API
colorFrom: teal
colorTo: red
sdk: docker
app_port: 7860
---

# RoBERTa AI Text Detector

A full-stack AI-text detector built from a fine-tuned RoBERTa sequence classification model. The app analyzes pasted text or text files, splits long documents into overlapping token windows, reports an overall Human/AI probability, and flags grouped regions with unusually high AI probability.

## Project Structure

```text
backend/   FastAPI inference API
frontend/  Vite React interface
model/     Fine-tuned RoBERTa artifact
notebooks/ Training and experiment notebooks
samples/   Demo text files
docs/      Model card and deployment notes
```

## Local Backend

```powershell
cd C:\Projects\PROJECTSEM7\backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 7860
```

Useful endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-file`

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
- Default suspicious region group size: 3 chunks
- Optional CPU dynamic quantization: `USE_QUANTIZED=true`

## Deployment

Use Hugging Face Spaces for the backend and Vercel for the frontend. See `docs/deployment.md`.

## Responsible Use

The model returns probabilistic signals, not proof. Use the output as one review aid, especially for long documents where chunk-level patterns may matter more than a single document-level label.
