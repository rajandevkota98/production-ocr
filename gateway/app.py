import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gateway.services.logger import get_logger
from gateway.services.exceptions import (
    OcrPlatformException,
    TaskNotFoundException,
    InvalidPayloadException,
    RedisConnectionException,
)
from gateway.sources.loader import detect_and_validate_format, encode_payload
from gateway.services.queue import QueueService

logger = get_logger("gateway_app")

queue_service: QueueService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue_service
    logger.info("Initializing OCR Gateway application...")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    queue_service = QueueService(redis_url=redis_url)
    logger.info("OCR Gateway ready.")
    yield
    logger.info("Shutting down OCR Gateway...")

app = FastAPI(
    title="Production OCR Platform",
    description="High-Throughput Visual Document Understanding Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

@app.exception_handler(OcrPlatformException)
async def custom_exception_handler(request: Request, exc: OcrPlatformException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "status_code": exc.status_code},
    )

class ProcessResponse(BaseModel):
    task_id: str
    status: str

@app.post("/process", status_code=202, response_model=ProcessResponse)
async def submit_document(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise InvalidPayloadException("File exceeds maximum allowed limit of 10MB.")

    extension = detect_and_validate_format(raw_bytes)
    task_id = str(uuid.uuid4())
    base64_data = encode_payload(raw_bytes)

    queue_service.enqueue_task(
        task_id=task_id,
        filename=file.filename or f"document.{extension}",
        extension=extension,
        base64_data=base64_data,
    )

    return ProcessResponse(task_id=task_id, status="queued")

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    return queue_service.get_task_status(task_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run("gateway.app:app", host="0.0.0.0", port=port, reload=False)
