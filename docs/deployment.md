# Deployment

## Backend on Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose Docker as the Space SDK.
3. Push the repository root, including `Dockerfile`, `backend/`, and `model/roberta_ai_detector_final/`.
4. Keep the default free CPU hardware first.
5. Set Space variables if needed:

```text
USE_QUANTIZED=true
CHUNK_TOKENS=512
OVERLAP_TOKENS=64
GROUP_SIZE=3
SUSPICIOUS_AI_THRESHOLD=85
SUSPICIOUS_DELTA_THRESHOLD=20
ALLOWED_ORIGINS=*
```

The Space exposes FastAPI on port `7860`. Free CPU Spaces can sleep when unused, so the frontend includes a backend readiness state.

For this project, the intended Space is:

```text
clashking/AIDetect
```

After logging in locally with `hf auth login`, deploy it with:

```powershell
cd C:\Projects\PROJECTSEM7
.\scripts\deploy_huggingface.ps1
```

## Frontend on Vercel

1. Import this repository into Vercel.
2. Set the Vercel project root directory to `frontend`.
3. Add the environment variable:

```text
VITE_API_URL=https://your-space-name.hf.space
```

4. Build command: `npm run build`
5. Output directory: `dist`

## Local Run

Backend:

```powershell
cd C:\Projects\PROJECTSEM7\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 7860
```

Frontend:

```powershell
cd C:\Projects\PROJECTSEM7\frontend
npm install
npm run dev
```

Open the Vite URL and analyze text through the FastAPI backend.
