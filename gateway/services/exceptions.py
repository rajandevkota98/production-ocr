class OcrPlatformException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class TaskNotFoundException(OcrPlatformException):
    def __init__(self, task_id: str):
        super().__init__(f"Task ID '{task_id}' not found", status_code=404)

class InvalidPayloadException(OcrPlatformException):
    def __init__(self, reason: str):
        super().__init__(f"Invalid payload: {reason}", status_code=400)

class RedisConnectionException(OcrPlatformException):
    def __init__(self, detail: str):
        super().__init__(f"Redis state store error: {detail}", status_code=502)
