# Production OCR Platform — Agent Instructions & Architectural Memory

Welcome, Agent. This repository contains the ground-up implementation of a **Production-Grade, Self-Scaling Visual Document Understanding (VDU) & OCR Pipeline**.

Whenever you work in this workspace, follow the instructions, architecture rules, and implementation patterns defined in this document.

---

## 1. System Architecture: 100% Cloud (Zero Credit Card Required)

The platform decouples ingestion, queue state, layout segmentation, and generative language model recognition into distinct, independently scalable cloud tiers requiring zero local hardware and zero credit card verification:

```mermaid
flowchart TD
    Client(["Client / AI Agent / Web User"]) -->|POST /process (PDF or Image)| Gateway["1. Ingestion Gateway (Hugging Face Spaces)\n• 2 vCPUs, 16 GB RAM (Free Tier - No Card)\n• Docker container on port 7860\n• Request validation & Base64 encoding"]
    
    Gateway --> Redis[("2. State & Task Store (Upstash Redis)\n• Free 10,000 commands/day\n• Hash task:{task_id}\n• List ocr_tasks")]
    
    subgraph ModalGPUTier ["Modal Serverless GPU Tier ($30/mo Free Credit)"]
        Layout["3. Stage 1: Layout Worker (Modal T4 GPU)\n• PP-DocLayoutV3 Segmentation\n• Bounding Box Cropping & Prompt Routing\n• Scale-to-zero in 60s"]
        
        VLLM["4. Stage 2: Generative SLM (Modal A10G GPU)\n• Qwen2-VL-2B-Instruct via vLLM\n• Continuous Batching & Prompt Specialization\n• Scale-to-zero in 120s"]
    end
    
    Gateway -->|Dispatch Job via Webhook / SDK| Layout
    Layout -->|Image crops + prompts| VLLM
    VLLM -->|Recognized text, tables, formulas| Layout
    Layout -->|Result JSON + Markdown| Redis
    
    Client -->|GET /status/{task_id}| Gateway
    Gateway -->|HGETALL task:{task_id}| Redis
```

---

## 2. Component Specifications

### A. Ingestion Gateway (Hugging Face Spaces)
- **Path**: `deploy/hf_spaces/` and `gateway/`
- **Host**: Hugging Face Spaces (Docker SDK).
- **Resources**: 2 vCPUs, 16 GB RAM (Free, 24/7, No Card Required).
- **Port**: Strictly listens on port `7860` as required by Hugging Face Spaces.
- **Endpoints**:
  - `POST /process`: Validates upload, stores in Redis, triggers Modal execution.
  - `GET /status/:task_id`: Queries task state and returns structured Markdown.
  - `GET /health`: Health probe.

### B. State & Buffer Store (Upstash Redis)
- **Host**: Configurable via `REDIS_URL` (Upstash serverless Redis `rediss://default:...@...upstash.io:6379`).
- **Data Structures**:
  - `task:{task_id}`: Hash storing task status (`"queued"`, `"processing"`, `"done"`, `"failed"`), input metadata, and final JSON output.
  - `ocr_tasks`: FIFO task queue.

### C. Layout & Inference Tier (Modal)
- **Path**: `inference/modal_app.py`
- **Modal App**: `modal.App("production-ocr")`
- **Volume**: `modal.Volume.from_name("ocr-models")` stores `PaddlePaddle/PP-DocLayoutV3` and `Qwen/Qwen3.5-4B` weights.
- **Stage 1 (Layout)**:
  - GPU: `T4` (lightweight, scale-to-zero after 60s idle).
  - Identifies bounding boxes, reading order, and semantic categories.
- **Stage 2 (Generative SLM)**:
  - GPU: `A10G` or `A100` (scale-to-zero after 120s idle).
  - Transcribes text, tables (Markdown), and formulas (LaTeX).

---

## 3. Directory Structure

```text
/home/rajandevkota/projects/production-ocr/
├── .agents/
│   └── rules/
│       └── ocr-pipeline-specs.md
├── AGENTS.md
├── GEMINI.md
├── .gitignore
├── deploy/
│   └── hf_spaces/           # Hugging Face Spaces deployment configuration
│       ├── Dockerfile
│       └── README.md
├── gateway/                 # Ingestion Gateway (FastAPI / Rust)
│   ├── app.py
│   ├── requirements.txt
│   ├── sources/
│   └── services/
├── inference/               # Modal GPU Functions & Model Serving
│   ├── modal_app.py
│   └── requirements.txt
└── tests/
    └── test_pipeline.py
```

---

## 4. Operational Runbook

### Deploying to Hugging Face Spaces
1. Create a Space at `huggingface.co/new-space` with Docker SDK (Blank template).
2. Set Space secret `MODAL_API_URL` and `REDIS_URL`.
3. Push `deploy/hf_spaces/` contents or set Space repo as git remote and push.

### Deploying to Modal
1. Authenticate Modal CLI: `modal setup`.
2. Deploy the serverless app: `modal deploy inference/modal_app.py`.
