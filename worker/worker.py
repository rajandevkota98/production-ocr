import os
import time
import json
import base64
import asyncio
import redis
import httpx
from loguru import logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "4"))
BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", "100"))
MODAL_ENDPOINT_URL = os.getenv("MODAL_ENDPOINT_URL", "")

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

async def process_task(task_id: str, client: httpx.AsyncClient):
    task_key = f"task:{task_id}"
    task_data = r.hgetall(task_key)
    if not task_data:
        logger.error(f"Task {task_id} not found in Redis.")
        return

    r.hset(task_key, "status", "processing")
    ext = task_data.get("extension", "pdf")
    temp_path = f"/dev/shm/{task_id}.{ext}"

    try:
        raw_b64 = task_data.get("data", "")
        file_bytes = base64.b64decode(raw_b64)

        # Write to shared memory RAM-disk
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        os.chmod(temp_path, 0o644)

        if MODAL_ENDPOINT_URL:
            # Dispatch to Modal serverless GPU endpoint
            res = await client.post(
                MODAL_ENDPOINT_URL,
                json={"task_id": task_id, "extension": ext, "data": raw_b64},
                timeout=300.0,
            )
            res.raise_for_status()
            modal_data = res.json()
            result_payload = modal_data.get("result", {})
        else:
            # Fallback mock result for local verification
            result_payload = {
                "markdown": f"# Document Content for {task_data.get('filename')}\n\nSuccessfully processed.",
                "layout": {"regions": []},
            }

        # Update Redis with result and clear payload
        r.hset(
            task_key,
            mapping={
                "status": "done",
                "result": json.dumps(result_payload),
                "data": "",
            },
        )
        logger.info(f"Task {task_id} completed successfully.")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        r.hset(task_key, mapping={"status": "failed", "error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def worker_loop():
    logger.info(f"Starting OCR Worker (Batch: {MAX_BATCH_SIZE}, Window: {BATCH_WINDOW_MS}ms)...")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                task_info = r.brpop("ocr_tasks", timeout=2)
                if not task_info:
                    continue

                _, first_task = task_info
                batch = [first_task]

                # Collector window to fill the batch
                deadline = time.time() + (BATCH_WINDOW_MS / 1000.0)
                while len(batch) < MAX_BATCH_SIZE:
                    timeout = max(0, deadline - time.time())
                    next_task = r.rpop("ocr_tasks")
                    if next_task:
                        batch.append(next_task)
                    elif timeout > 0:
                        await asyncio.sleep(0.01)
                    else:
                        break

                # Process batch concurrently
                await asyncio.gather(*[process_task(t, client) for t in batch])

            except Exception as e:
                logger.error(f"Worker loop encountered an error: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(worker_loop())
