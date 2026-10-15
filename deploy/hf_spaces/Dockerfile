FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY gateway/requirements.txt requirements.txt
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY gateway /app/gateway

# Hugging Face Spaces strictly requires port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "gateway.app"]
