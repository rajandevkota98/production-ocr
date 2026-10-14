# 🤖 Production OCR Platform — Agent Instructions & Architectural Memory

Welcome, Agent. This repository contains the ground-up implementation of a **Production-Grade, Self-Scaling Visual Document Understanding (VDU) & OCR Pipeline**.

Whenever you work in this workspace, follow the instructions, architecture rules, and implementation patterns defined in this document.

---

## 🏛️ System Architecture

The platform decouples ingestion, queue state, layout segmentation, and generative language model recognition into distinct, independently scalable tiers:

```mermaid
flowchart TD
    Client(["Client / AI Agent / Claude Code"]) -->|POST /process (PDF or Image)| Gateway["1. Ingestion Gateway (Rust Axum)\n• 10MB limit enforcement\n• Multipart parsing\n• Base64 encoding\n• Atomic HSET + LPUSH"]
    
    Gateway --> Redis[("2. State & Task Store (Redis)\n• Hash task:{task_id}\n• List ocr_tasks")]
    
    subgraph LayoutWorkerTier ["Layout Worker Tier (NVIDIA T4 / GPU)"]
        Worker["3. Consumer Worker (Python)\n• Dynamic Batch Collector (100ms window, max 4)\n• Zero-Copy RAM Disk (/dev/shm)\n• PP-DocLayoutV3 Segmentation\n• Bounding Box Cropping & Prompt Routing"]
    end
    
    Redis -->|BRPOP / RPOP FIFO| Worker
    
    subgraph VLLMInferenceTier ["Generative Inference Tier (NVIDIA A100 / GPU)"]
        vLLM["4. vLLM Inference Server\n• Qwen 3.5 (4B / 2B)\n• Continuous Batching (MAX_NUM_SEQS=512)\n• Chunked Prefill (MAX_NUM_BATCHED_TOKENS=262144)\n• Multi-Token Prediction (MTP)"]
    end
    
    Worker -->|POST /v1/chat/completions| vLLM
    vLLM -->|Text / Table / Formula / Chart desc| Worker
    Worker -->|HSET status=done, result=JSON, data=''| Redis
    
    Client -->|GET /status/{task_id}| Gateway
    Gateway -->|HGETALL task:{task_id}| Redis
    
    subgraph AgentIntegration ["Agentic Tooling"]
        MCP["5. FastMCP Server\n• @mcp.tool parse_document()\n• Workspace boundary path check\n• Stdio & SSE transports"]
    end
    MCP --> Gateway
```

---

## 📜 Key Architectural Directives & Rules

### 1. Ingestion Gateway (Rust + Axum)
- **Path**: `gateway/`
- **Responsibilities**:
  - Expose `POST /process`, `GET /status/:task_id`, and `GET /health`.
  - Enforce a 10MB upload limit using `tower_http::limit::RequestBodyLimitLayer`.
  - Generate UUID v4 `task_id` for each request.
  - Write state atomically to Redis using a single `HSET` command:
    - Key: `task:{task_id}`
    - Fields: `status: "queued"`, `filename`, `extension`, `data: <base64>`
  - Push `task_id` to Redis list `ocr_tasks` using `LPUSH`.
  - Return HTTP `202 Accepted` with JSON `{ "task_id": "...", "status": "queued" }`.
  - Return task status with HTTP `200` from `GET /status/:task_id` parsing the Redis `result` field.

### 2. State & Buffer Store (Redis)
- **Host**: Configurable via `REDIS_HOST` (default `ocr-redis-service` or `localhost`).
- **Port**: `REDIS_PORT` (default `6379`).
- **Data Structures**:
  - `task:{task_id}`: Hash containing task metadata, status, payload, and finished result.
  - `ocr_tasks`: List acting as the FIFO queue (`LPUSH` on ingestion, `BRPOP` + `RPOP` on worker consumption).

### 3. Asynchronous Consumer Worker (Python + GLM-OCR / PP-DocLayoutV3)
- **Path**: `worker/`
- **Dynamic Batch Collector**:
  - Block on queue using `brpop("ocr_tasks", timeout=2)`.
  - Once a task arrives, wait up to `BATCH_WINDOW_MS=100` ms to collect up to `MAX_BATCH_SIZE=4` tasks using `rpop` to preserve strict FIFO ordering.
- **Zero-Copy RAM Handoff**:
  - Decode Base64 payloads directly into `/dev/shm/{task_id}.{ext}`.
  - Apply `os.chmod(temp_path, 0o644)` to ensure child processes can read the shared memory buffer.
- **Layout Segmentation**:
  - Run `PP-DocLayoutV3` over the batch.
  - Segment document into semantic regions: `text`, `table`, `formula`, `chart`.
  - Abandon non-content regions (headers, footers, page numbers) to avoid context pollution.
  - Map each region to its specialized prompt.
- **Cleanup**:
  - Update Redis with `status="done"` and the structured JSON/Markdown result.
  - Clear the large Base64 `data` string from Redis to prevent memory bloat.
  - Delete temporary files in `/dev/shm/` inside a `finally` block.

### 4. Generative Inference Engine (vLLM)
- **Path**: `server/` or `inference/`
- **Model**: `Qwen/Qwen3.5-4B` (or `Qwen/Qwen3.5-2B` for lighter hardware).
- **Critical vLLM Launch Parameters**:
  ```bash
  vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --max-num-batched-tokens 262144 \
    --max-num-seqs 512 \
    --limit-mm-per-prompt '{"image":1, "video":0}' \
    --mm-encoder-tp-mode data \
    --load-format instanttensor \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-chunked-prefill
  ```
  - `MAX_NUM_BATCHED_TOKENS=262144`: Crucial for handling multi-crop document batches without prefill choking.
  - `MAX_MODEL_LEN=16384`: Optimal balance for 6k vision tokens + 8k generation tokens.

### 5. Agentic MCP Integration (FastMCP)
- **Path**: `mcp/`
- Expose `@mcp.tool() parse_document(file_path: str, include_layout: bool = False)`.
- Enforce strict workspace confinement (`path.is_relative_to(WORKSPACE)`) to prevent path traversal vulnerabilities.
- Support both `stdio` and `sse` transport mechanisms.

### 6. Event-Driven Autoscaling (KEDA)
- **Worker Scaler**: Scale on Redis list length `ocr_tasks` (`threshold = 1`) + cron warm-start during business hours.
- **vLLM Scaler**: Scale on Prometheus metric `vllm:num_requests_waiting` (`threshold = 1`).
- **API Scaler**: Scale on CPU utilization (`> 70%`).

---

## 🗂️ Target Repository Structure

```text
/home/rajandevkota/projects/production-ocr/
├── .agents/
│   └── rules/
│       └── ocr-pipeline-specs.md
├── AGENTS.md
├── GEMINI.md
├── Makefile
├── docker-compose.yml
├── gateway/                 # Rust Axum Producer API Gateway
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       └── main.rs
├── worker/                  # Python Dynamic Batching Consumer
│   ├── Dockerfile
│   ├── config.yaml
│   ├── requirements.txt
│   └── worker.py
├── inference/               # vLLM Model Serving Configuration
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── requirements.txt
├── mcp/                     # FastMCP Server for AI Agents
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py
├── deploy/                  # Kubernetes (AKS / GKE) & Terraform
│   ├── k8s/
│   │   ├── apps/
│   │   └── keda/
│   └── terraform/
└── tests/
    └── test_pipeline.py
```

---

## 🛠️ Ground-Up Development Roadmap

1. **Phase 1: Local Docker Compose Stack**
   - Stand up Redis.
   - Build & run the Rust Gateway (`gateway/`).
   - Wire mock or containerized inference worker.
2. **Phase 2: Worker & Layout Pipeline**
   - Implement `worker.py` with dynamic collector (`100ms` window, `max 4`).
   - Implement `/dev/shm` zero-copy transfer.
   - Configure GLM-OCR SDK / PP-DocLayoutV3 with prompt mappings.
3. **Phase 3: vLLM Inference Container**
   - Build optimized `entrypoint.sh` with chunked prefill and token settings.
   - Connect worker HTTP requests to vLLM `/v1/chat/completions`.
4. **Phase 4: FastMCP Agent Tool**
   - Create MCP tool for Claude Code & Antigravity CLI.
5. **Phase 5: Cloud Deployment & KEDA Scaling**
   - Deploy onto Kubernetes (AKS / GKE) with node taints for T4 and A100 GPUs.
