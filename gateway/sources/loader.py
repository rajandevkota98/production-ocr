import base64
from typing import Tuple
from gateway.services.exceptions import InvalidPayloadException

def detect_and_validate_format(raw_bytes: bytes) -> str:
    if len(raw_bytes) < 4:
        raise InvalidPayloadException("File size is too small to determine format.")
    
    if raw_bytes.startswith(b"%PDF-"):
        return "pdf"
    elif raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif raw_bytes.startswith(b"\xff\xd8\xff"):
        return "jpg"
    elif len(raw_bytes) >= 12 and raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "webp"
    
    raise InvalidPayloadException("Unsupported file type. Expected PDF, PNG, JPEG, or WebP.")

def encode_payload(raw_bytes: bytes) -> str:
    return base64.b64encode(raw_bytes).decode("utf-8")
