FROM python:3.11-slim

# 시스템 의존성
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY server.py index.html ./

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# 🌟 FastAPI 서버(uvicorn)를 7860 포트에서 실행
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]