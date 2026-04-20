FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/roberta_ai_detector_final
ENV USE_QUANTIZED=true
ENV PORT=7860

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY model /app/model

WORKDIR /app/backend

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
