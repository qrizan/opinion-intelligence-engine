# base image - Python 3.12 slim (lebih kecil)
FROM python:3.12-slim

# set working directory
WORKDIR /app

# install system dependencies (untuk beberapa Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first (untuk layer caching)
COPY requirements.txt .

# install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# build arguments untuk folder IDs (dari .env atau build command)
ARG SENTIMENT_FOLDER_ID
ARG SUMMARIZATION_FOLDER_ID

# set environment variables untuk download_models.py
ENV SENTIMENT_FOLDER_ID=${SENTIMENT_FOLDER_ID}
ENV SUMMARIZATION_FOLDER_ID=${SUMMARIZATION_FOLDER_ID}

# download models dari Google Drive (saat build)
RUN python3 download_models.py

# expose port untuk FastAPI/Gradio
EXPOSE 8000

# health check (opsional, untuk Render)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# run server
CMD ["python3", "api/main.py"]