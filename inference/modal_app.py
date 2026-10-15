import os
import io
import base64
from typing import Dict, Any, List
import modal

app = modal.App("production-ocr")

models_volume = modal.Volume.from_name("ocr-models", create_if_missing=True)

# Image definition with required system and Python OCR libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "poppler-utils", "tesseract-ocr")
    .pip_install(
        "fastapi[standard]>=0.110.0",
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "pytesseract>=0.3.10",
        "redis>=5.0.0",
        "loguru>=0.7.2",
        "pdf2image>=1.17.0",
        "PyMuPDF>=1.24.0",
        "Pillow>=10.2.0",
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
    )
)

PROMPTS = {
    "text": "OCR and recognize the text in this image. Maintain the original structure and formatting. Output only the recognized text:",
    "table": "Convert this table image into a clean Markdown table format. Ensure all rows and columns are preserved accurately. Only output the markdown table:",
    "formula": "Convert this mathematical formula into LaTeX format. Wrap the result in appropriate delimiters. Only output the LaTeX string:",
    "image": "Describe this image in detail. If it is a chart or graph, explain the axes, data points, and overall trend. Provide a clear summary:",
}

@app.function(
    image=image,
    gpu="T4",
    volumes={"/mnt/models": models_volume},
    scaledown_window=60,
    timeout=300,
)
def parse_layout(file_bytes: bytes, extension: str) -> Dict[str, Any]:
    """
    Stage 1: Layout & Text Extraction on T4 GPU.
    Performs block segmentation, bounding box extraction, and OCR.
    """
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    regions_to_process = []
    total_pages = 1

    try:
        if extension.lower() == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                blocks = page.get_text("blocks")
                page_text_found = False

                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    x0, y0, x1, y1, text, block_no, b_type = b
                    cleaned = text.strip()
                    if cleaned:
                        page_text_found = True
                        regions_to_process.append({
                            "page": page_idx + 1,
                            "label": "table" if "\t" in cleaned or "|" in cleaned else "text",
                            "bbox": [round(x0), round(y0), round(x1), round(y1)],
                            "content": cleaned,
                        })

                # If scanned PDF with no text layer, rasterize and run OCR
                if not page_text_found:
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        regions_to_process.append({
                            "page": page_idx + 1,
                            "label": "text",
                            "bbox": [0, 0, img.width, img.height],
                            "content": ocr_text,
                        })
        else:
            # Single image OCR
            img = Image.open(io.BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(img).strip()
            if not ocr_text:
                ocr_text = "No readable text detected in image."
            
            regions_to_process.append({
                "page": 1,
                "label": "text",
                "bbox": [0, 0, img.width, img.height],
                "content": ocr_text,
            })

    except Exception as e:
        regions_to_process.append({
            "page": 1,
            "label": "error",
            "bbox": [],
            "content": f"Processing error: {str(e)}",
        })

    return {
        "total_pages": total_pages,
        "regions": regions_to_process,
        "status": "layout_complete",
    }

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/mnt/models": models_volume},
    scaledown_window=120,
    timeout=300,
)
def transcribe_crops(crops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stage 2: Formatting and Structured Markdown Assembly.
    """
    results = []
    markdown_lines = []

    for crop in crops:
        content = crop.get("content", "").strip()
        label = crop.get("label", "text")
        page = crop.get("page", 1)

        if content:
            results.append({
                "page": page,
                "label": label,
                "bbox": crop.get("bbox", []),
                "content": content,
            })
            markdown_lines.append(content)

    final_markdown = "\n\n".join(markdown_lines) if markdown_lines else "No text recognized."
    return {
        "markdown": final_markdown,
        "layout": {"regions": results},
        "status": "done",
    }

@app.function(
    image=image,
    timeout=600,
)
@modal.fastapi_endpoint(method="POST")
def process_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified HTTP Webhook for Hugging Face Spaces Gateway.
    Accepts task payload, coordinates Stage 1 and Stage 2, and returns results.
    """
    raw_b64 = payload.get("data", "")
    extension = payload.get("extension", "pdf")
    file_bytes = base64.b64decode(raw_b64)

    # 1. Run Layout & Text Extraction on T4
    layout_result = parse_layout.remote(file_bytes, extension)
    regions = layout_result.get("regions", [])

    # 2. Assemble Structured Markdown Output
    ocr_result = transcribe_crops.remote(regions)

    return {
        "task_id": payload.get("task_id", ""),
        "status": "done",
        "result": {
            "markdown": ocr_result.get("markdown", ""),
            "layout": ocr_result.get("layout", {}),
        },
    }
