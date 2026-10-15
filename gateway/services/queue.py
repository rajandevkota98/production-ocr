import json
import os
from typing import Optional, Dict, Any
import redis
from gateway.services.exceptions import RedisConnectionException, TaskNotFoundException
from gateway.services.logger import get_logger

logger = get_logger("queue_service")

class QueueService:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise RedisConnectionException(str(e))

    def enqueue_task(self, task_id: str, filename: str, extension: str, base64_data: str) -> None:
        try:
            task_key = f"task:{task_id}"
            self.client.hset(
                task_key,
                mapping={
                    "status": "queued",
                    "filename": filename,
                    "extension": extension,
                    "data": base64_data,
                },
            )
            self.client.lpush("ocr_tasks", task_id)
            logger.info(f"Task {task_id} queued successfully for {filename}")
        except Exception as e:
            logger.error(f"Redis enqueue failed for {task_id}: {e}")
            raise RedisConnectionException(str(e))

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        try:
            task_key = f"task:{task_id}"
            data = self.client.hgetall(task_key)
            if not data:
                raise TaskNotFoundException(task_id)

            status = data.get("status", "unknown")
            raw_result = data.get("result")
            result = json.loads(raw_result) if raw_result else None
            error = data.get("error")

            return {
                "task_id": task_id,
                "status": status,
                "result": result,
                "error": error,
            }
        except TaskNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Redis fetch failed for {task_id}: {e}")
            raise RedisConnectionException(str(e))
