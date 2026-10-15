#!/usr/bin/env python3
import os
import sys
import time
import pathlib
import asyncio
from typing import Dict, Any
import httpx
from mcp.server.fastmcp import FastMCP

OCR_GATEWAY_URL = os.getenv("OCR_GATEWAY_URL", "http://localhost:7860")
WORKSPACE = pathlib.Path(os.getenv("OCR_WORKSPACE", ".")).resolve()
POLL_INTERVAL_SEC = 0.5
MAX_TIMEOUT_SEC = 120.0

mcp = FastMCP(
    name="production-ocr-tool",
    instructions="High-throughput visual document understanding tool for analyzing PDFs, images, and tables.",
)

def _validate_path(file_path: str) -> pathlib.Path:
    target = (WORKSPACE / file_path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"Path escapes workspace boundaries: {file_path}")
    if not target.is_file():
        raise ValueError(f"Document file not found: {file_path}")
    return target

@mcp.tool()
async def parse_document(file_path: str, include_layout: bool = False) -> Dict[str, Any]:
    """
    Submits a document (PDF or image) to the Production OCR Platform and retrieves structured Markdown.
    
    Args:
        file_path: Path to the PDF or image file relative to the workspace.
        include_layout: Whether to include detected layout regions and bounding boxes.
    """
    try:
        target = _validate_path(file_path)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            with open(target, "rb") as f:
                res = await client.post(
                    f"{OCR_GATEWAY_URL.rstrip('/')}/process",
                    files={"file": (target.name, f)},
                )
                res.raise_for_status()
                task_data = res.json()
        except Exception as e:
            return {"success": False, "error": f"Failed to submit task: {e}"}

        task_id = task_data.get("task_id")
        if not task_id:
            return {"success": False, "error": "No task_id received from gateway."}

        # Poll status until complete
        start = time.time()
        status_url = f"{OCR_GATEWAY_URL.rstrip('/')}/status/{task_id}"
        while (time.time() - start) < MAX_TIMEOUT_SEC:
            try:
                res = await client.get(status_url)
                if res.status_code == 200:
                    data = res.json()
                    status = data.get("status")
                    if status == "done":
                        result = data.get("result", {})
                        payload = {
                            "success": True,
                            "task_id": task_id,
                            "markdown": result.get("markdown", ""),
                            "elapsed_sec": round(time.time() - start, 2),
                        }
                        if include_layout:
                            payload["layout"] = result.get("layout", {})
                        return payload
                    elif status == "failed":
                        return {"success": False, "task_id": task_id, "error": data.get("error")}
                await asyncio.sleep(POLL_INTERVAL_SEC)
            except Exception:
                await asyncio.sleep(POLL_INTERVAL_SEC)

        return {"success": False, "error": f"Timed out after {MAX_TIMEOUT_SEC} seconds"}

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "sse" or "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
