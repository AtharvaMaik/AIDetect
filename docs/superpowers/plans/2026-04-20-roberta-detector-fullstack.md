# RoBerta Detector Fullstack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing fine-tuned RoBERTa notebook project into a deployable full-stack AI text detector using only the user's trained model.

**Architecture:** A FastAPI backend loads the local Hugging Face model once, optionally applies CPU dynamic quantization, analyzes long text through overlapping token chunks, groups adjacent chunks into suspicious regions, and returns structured JSON. A Vite React frontend calls the backend and presents smooth document-level, chunk-level, and suspicious-region results. Deployment targets are Vercel for the frontend and Hugging Face Spaces free CPU for the backend.

**Tech Stack:** Python 3.10, FastAPI, PyTorch, Transformers, Pydantic, pytest, React, Vite, TypeScript, CSS, Hugging Face Spaces Docker.

---

### File Structure

- `backend/app/main.py`: FastAPI routes and CORS setup.
- `backend/app/config.py`: model path, inference defaults, and environment settings.
- `backend/app/model_service.py`: tokenizer/model loading, optional quantization, and prediction orchestration.
- `backend/app/chunking.py`: token-window chunking and grouped suspicious-region detection.
- `backend/app/schemas.py`: request and response models.
- `backend/tests/test_chunking.py`: deterministic tests for chunk grouping and aggregation.
- `backend/requirements.txt`: backend runtime/test dependencies.
- `backend/Dockerfile`: Hugging Face Spaces backend container.
- `frontend/src/App.tsx`: main detector UI.
- `frontend/src/api.ts`: backend API client.
- `frontend/src/types.ts`: shared response types.
- `frontend/src/styles.css`: responsive polished interface.
- `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/index.html`: frontend project config.
- `docs/deployment.md`: Vercel and Hugging Face Spaces deployment steps.
- `docs/model-card.md`: model details, limitations, and responsible-use notes.
- `README.md`: project overview and local run guide.

### Tasks

- [ ] Clean workspace by moving final model to `model/roberta_ai_detector_final`, moving notebooks to `notebooks/`, moving `your.txt` to `samples/constitution.txt`, and deleting `.ipynb_checkpoints` plus `roberta_ai_detector/checkpoint-*`.
- [ ] Add backend schemas, chunking, grouping, model service, and API routes.
- [ ] Add backend tests for long-text chunk aggregation and suspicious-region detection.
- [ ] Add frontend Vite React app with text input, upload, result summary, model confidence, chunk region cards, and deployment-aware API URL.
- [ ] Add Dockerfile, requirements, README, model card, and deployment docs.
- [ ] Verify backend tests, frontend build, and local API startup.

### Verification

- Run `python -m pytest backend/tests -q` and expect all tests to pass.
- Run `npm --prefix frontend install` then `npm --prefix frontend run build` and expect a production build.
- Run `python -m uvicorn app.main:app --host 127.0.0.1 --port 7860` from `backend/` and check `/health`.
- Run the frontend dev server and confirm the UI loads.

### Self-Review

- Scope uses only the user's fine-tuned RoBERTa model.
- The backend supports quantized CPU inference but keeps original weights as source of truth.
- The app reports whole-document results and unusually high-AI grouped regions.
- No external model ensemble or retraining is included in this implementation.
