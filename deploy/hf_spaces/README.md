# Deploying Gateway to Hugging Face Spaces (Free Cloud Tier)

This gateway provides the public API endpoint for document ingestion, queuing tasks into Redis and triggering Modal GPU inference.

## Prerequisites
1. Free Hugging Face account at [huggingface.co](https://huggingface.co) (no credit card needed).
2. Free Upstash Redis database at [upstash.com](https://upstash.com) (no credit card needed).

## Steps to Deploy

1. Create a new Space:
   - Navigate to **Hugging Face > New Space**.
   - Set Space Name: `production-ocr-gateway`.
   - License: `MIT`.
   - Select SDK: **Docker** > **Blank Template**.
   - Hardware: **2 vCPU · 16 GB RAM (Free)**.

2. Configure Secrets in Space Settings:
   - Go to **Settings > Variables and Secrets**.
   - Add Secret `REDIS_URL`: `rediss://default:<password>@<host>:6379` (from Upstash).

3. Push Code to Space:
   - Follow instructions on the Space page to push the repository or upload the files (`Dockerfile`, `gateway/`).
