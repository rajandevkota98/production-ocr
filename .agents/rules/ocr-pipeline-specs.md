# Production OCR Pipeline Technical Rules & Specifications

This specification serves as the persistent rulebook for any agent implementing or modifying components in this repository.

## 1. Cloud Network & API Specifications

### Hugging Face Spaces Gateway
- **Port Requirement**: The gateway service **MUST** bind to `0.0.0.0:7860` (or honor the `PORT` environment variable which defaults to `7860`). Hugging Face Spaces ingress routes traffic only to port `7860`.
- **Endpoint**: `POST /process`
  - Input: `multipart/form-data` with form field `file`.
  - Body Limit: Up to 10MB (`10 * 1024 * 1024` bytes).
  - Validation: Verify file magic bytes (`%PDF-`, PNG, JPEG, WebP).
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

## 2. Upstash Redis Schema & Configuration

- **Connection URL**: Supports TLS connections (`rediss://default:<password>@<host>:6379`).
- **Key**: `task:{task_id}` (Hash)
  - `status`: String (`"queued"`, `"processing"`, `"done"`, `"failed"`)
  - `filename`: Original upload filename
  - `extension`: Detected file extension (`"pdf"`, `"png"`, `"jpg"`)
  - `data`: Base64 encoded binary payload (cleared to `""` upon completion to save memory)
  - `result`: Serialized JSON string containing `{"markdown": "...", "layout": {...}}`
  - `error`: Error message string if failed

## 3. Modal GPU Execution Rules

- **Scale-to-Zero Constraints**:
  - `parse_layout` (Stage 1): Run on `gpu="T4"` with `scaledown_window=60`.
  - `transcribe_crops` (Stage 2): Run on `gpu="A10G"` (or `A100`) with `scaledown_window=120`.
- **Model Storage**:
  - Pre-load and persist models into a named `modal.Volume("ocr-models")` mounted at `/mnt/models` to prevent cold downloads from Hugging Face on every invocation.
