FROM python:3.11-slim

LABEL maintainer="Deepak Singh <todeepaksingh.com@gmail.com>"
LABEL description="Early Detection of Depression from Text Using NLP"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e . --no-deps

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "depression_detector.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
