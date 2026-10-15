import os
import io
import base64
from typing import Dict, Any, List
from PIL import Image
import modal

app = modal.App("production-ocr")

models_volume = modal.Volume.from_name("ocr-models", create_if_missing=True)

# Image definition with real layout detection and Qwen Vision SLM dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "poppler-utils")
    .pip_install(
        "fastapi[standard]>=0.110.0",
        "torch>=2.2.0",
        "transformers>=4.45.0",
        "accelerate>=0.30.0",
        "qwen-vl-utils>=0.0.8",
        "rapid_layout>=0.1.4",
        "rapidocr_onnxruntime>=1.4.0",
        "PyMuPDF>=1.24.0",
        "Pillow>=10.2.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
    )
)

PROMPTS = {
    "text": "OCR and recognize the text in this image crop. Maintain the original reading flow and formatting. Output only the recognized text:",
    "table": "Convert this table image crop into a clean, valid GitHub-flavored Markdown table. Preserve all columns, rows, headers, and numerical alignment accurately. Only output the markdown table:",
    "formula": "Convert this mathematical formula into clean LaTeX format. Wrap display formulas in $$ and inline formulas in $. Only output the LaTeX string:",
    "figure": "Describe this visual figure, chart, or diagram accurately in 1-2 concise sentences. If it is a graph, summarize axes and trends. Output only the description:",
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
    Stage 1: PP-DocLayoutV3 Semantic Layout Detection on NVIDIA T4.
    Segments document pages into semantic blocks (text, table, formula, figure)
    and extracts high-resolution image crops for Stage 2 generative recognition.
    """
    import fitz  # PyMuPDF
    import numpy as np
    from rapid_layout import RapidLayout

    # Initialize PP-DocLayoutV3 engine (PicoDet / YOLOv8 layout model)
    try:
        layout_engine = RapidLayout(conf_thres=0.3)
    except Exception:
        layout_engine = None

    pages_crops = []
    total_pages = 1

    try:
        # Load PDF or raster image
        if extension.lower() == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            pil_images = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                pil_images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        else:
            pil_images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
            total_pages = len(pil_images)

        # Process each page with layout detection
        for page_idx, page_img in enumerate(pil_images):
            img_np = np.array(page_img)
            boxes, scores, class_names = [], [], []

            if layout_engine is not None:
                try:
                    layout_res, _ = layout_engine(img_np)
                    if layout_res:
                        for item in layout_res:
                            # item: [bbox, label, score]
                            bbox = item[0]
                            label = str(item[1]).lower()
                            score = float(item[2])
                            boxes.append(bbox)
                            scores.append(score)
                            class_names.append(label)
                except Exception:
                    pass

            # Fallback if no layout boxes found: treat as full page text block
            if not boxes:
                boxes = [[0, 0, page_img.width, page_img.height]]
                class_names = ["text"]
                scores = [1.0]

            # Crop semantic regions and serialize
            for b_idx, (box, label, score) in enumerate(zip(boxes, class_names, scores)):
                x0, y0, x1, y1 = [int(v) for v in box]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(page_img.width, x1), min(page_img.height, y1)

                if (x1 - x0) < 10 or (y1 - y0) < 10:
                    continue

                crop_pil = page_img.crop((x0, y0, x1, y1))
                buf = io.BytesIO()
                crop_pil.save(buf, format="JPEG", quality=95)
                crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                # Map detection label to specialized category
                cat = "text"
                if "table" in label:
                    cat = "table"
                elif "formula" in label or "equation" in label:
                    cat = "formula"
                elif "fig" in label or "image" in label or "chart" in label:
                    cat = "figure"

                pages_crops.append({
                    "page": page_idx + 1,
                    "region_id": b_idx,
                    "label": cat,
                    "score": round(score, 3),
                    "bbox": [x0, y0, x1, y1],
                    "crop_b64": crop_b64,
                })

    except Exception as e:
        pages_crops.append({
            "page": 1,
            "region_id": 0,
            "label": "error",
            "score": 0.0,
            "bbox": [],
            "crop_b64": "",
            "error": str(e)
        })

    return {
        "total_pages": total_pages,
        "regions": pages_crops,
        "status": "layout_complete",
    }


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/mnt/models": models_volume},
    scaledown_window=120,
    timeout=600,
)
class QwenSLMTranscriber:
    """
    Stage 2: Targeted Multimodal Generative OCR using Qwen Vision-Language SLM.
    Loads Qwen2-VL / Qwen 3.5 on NVIDIA A10G and generates structured Markdown/LaTeX.
    """
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"Loading {self.model_id} on NVIDIA A10G GPU...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()
        print(f"{self.model_id} initialized successfully.")

    @modal.method()
    def transcribe_crops(self, crops: List[Dict[str, Any]]) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info
        import torch

        results = []
        markdown_sections = []

        for crop_meta in crops:
            b64_crop = crop_meta.get("crop_b64", "")
            label = crop_meta.get("label", "text")
            page = crop_meta.get("page", 1)
            bbox = crop_meta.get("bbox", [])

            if not b64_crop:
                continue

            try:
                crop_bytes = base64.b64decode(b64_crop)
                crop_pil = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
                prompt_text = PROMPTS.get(label, PROMPTS["text"])

                # Build multimodal chat template for Qwen2-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": crop_pil},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.1,
                        top_p=0.9,
                        repetition_penalty=1.05,
                    )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

            except Exception as e:
                output_text = f"Transcription error: {str(e)}"

            results.append({
                "page": page,
                "label": label,
                "bbox": bbox,
                "content": output_text,
            })
            markdown_sections.append(output_text)

        final_markdown = "\n\n".join(markdown_sections) if markdown_sections else "No text recognized."
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
    Stage 1 (Layout on T4) -> Stage 2 (Qwen SLM on A10G) -> Structured Markdown.
    """
    raw_b64 = payload.get("data", "")
    extension = payload.get("extension", "pdf")
    file_bytes = base64.b64decode(raw_b64)

    # 1. Execute Stage 1 Layout Detection on T4 GPU
    layout_result = parse_layout.remote(file_bytes, extension)
    regions = layout_result.get("regions", [])

    # 2. Execute Stage 2 Generative Vision SLM Transcription on A10G GPU
    transcriber = QwenSLMTranscriber()
    ocr_result = transcriber.transcribe_crops.remote(regions)

    return {
        "task_id": payload.get("task_id", ""),
        "status": "done",
        "result": {
            "markdown": ocr_result.get("markdown", ""),
            "layout": ocr_result.get("layout", {}),
        },
    }
