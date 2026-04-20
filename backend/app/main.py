from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .model_service import ModelService
from .schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse


settings = get_settings()
app = FastAPI(title=settings.api_title)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService(settings)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_path_exists=settings.model_path.exists(),
        quantization_enabled=settings.use_quantized,
    )


@app.get("/model-info")
def model_info() -> dict:
    if not settings.model_path.exists():
        raise HTTPException(status_code=500, detail=f"Model path not found: {settings.model_path}")
    return get_model_service().info().__dict__


@app.post("/predict", response_model=AnalyzeResponse)
def predict(request: AnalyzeRequest) -> dict:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty after trimming whitespace.")
    if len(text) > settings.max_characters:
        raise HTTPException(
            status_code=413,
            detail=f"Text is too large. Maximum is {settings.max_characters} characters.",
        )

    chunk_tokens = request.chunk_tokens or settings.default_chunk_tokens
    overlap_tokens = request.overlap_tokens or settings.default_overlap_tokens
    group_size = request.group_size or settings.default_group_size

    try:
        return get_model_service().analyze(text, chunk_tokens, overlap_tokens, group_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-file", response_model=AnalyzeResponse)
async def predict_file(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Only UTF-8 text files are supported.") from exc

    return predict(AnalyzeRequest(text=text))
