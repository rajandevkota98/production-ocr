# Production OCR Pipeline Technical Rules & Specifications

This specification serves as the persistent rulebook for any agent implementing or modifying components in this repository.

## 1. Network & API Specifications

### Rust Gateway
- **Endpoint**: `POST /process`
  - Input: `multipart/form-data` with form field `file`.
  - Body Limit: Strictly enforced at `10 * 1024 * 1024` bytes (10MB).
  - Validation: Ensure MIME magic bytes match supported formats (PDF `%PDF-`, PNG, JPEG, WebP).
  - Return: `202 Accepted` with payload:
    ```json
    {
      "task_id": "uuid-v4-string",
      "status": "queued"
    }
    ```
- **Endpoint**: `GET /status/:task_id`
  - Query Redis: `HGETALL task:{task_id}`
  - Missing Key: Returns `404 Not Found` with `{ "error": "Task ID not found" }`.
  - Present Key: Return `200 OK` with payload:
    ```json
    {
      "task_id": "uuid-v4-string",
      "status": "queued | processing | done | failed",
      "result": { "markdown": "...", "layout": { ... } },
      "error": null
    }
    ```

## 2. Redis Key & State Schema

- **Key**: `task:{task_id}` (Hash)
  - `status`: String (`"queued"`, `"processing"`, `"done"`, `"failed"`)
  - `filename`: Original upload filename
  - `extension`: Detected file extension (`"pdf"`, `"png"`, `"jpg"`)
  - `data`: Base64 encoded binary payload (cleared to `""` upon completion)
  - `result`: Serialized JSON string containing `{"markdown": "...", "layout": {...}}`
  - `error`: Error message string if failed

- **Queue**: `ocr_tasks` (List)
  - Ingestion: Producer performs `LPUSH ocr_tasks {task_id}`
  - Consumption: Worker performs `BRPOP ocr_tasks` followed by non-blocking `RPOP ocr_tasks` within `BATCH_WINDOW_MS` to maintain strict FIFO.

## 3. Worker Dynamic Batching & Memory Safety Rules

- **FIFO Guarantee**: Never use `LPOP` with an `LPUSH` queue; always use `RPOP` to prevent queue inversion and task starvation under load.
- **Shared Memory RAM-Disk**:
  - Always write task data to `/dev/shm/{task_id}.{ext}`.
  - Always set file permissions to `0o644` (`os.chmod`).
  - Always unlink/remove temporary files in a `finally:` block.
- **Batch Window Tuning**:
  - `MAX_BATCH_SIZE`: Default `4` (matches T4 GPU layout batching).
  - `BATCH_WINDOW_MS`: Default `100` ms.

## 4. vLLM Engine Tuning Rules

- **Prefill Bottleneck Elimination**:
  - `MAX_NUM_BATCHED_TOKENS` must be set to at least `262144`. Each multi-page crop prompt can expand to ~6,000 vision tokens; lower limits throttle throughput.
- **Context Length**:
  - Set `MAX_MODEL_LEN=16384` to prevent excessive VRAM allocation while comfortably handling large multi-crop requests.
- **Chunked Prefill**:
  - Always pass `--enable-chunked-prefill` to co-schedule prefill chunks with active decoding steps and prevent latency spikes.
