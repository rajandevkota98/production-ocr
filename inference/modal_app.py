import os
import io
import base64
from typing import Dict, Any, List
from PIL import Image
import modal

app = modal.App("production-ocr")

models_volume = modal.Volume.from_name("ocr-models", create_if_missing=True)

# Image definition with vLLM, PP-DocLayoutV3, and PyTorch CUDA dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "poppler-utils")
    .pip_install(
        "vllm>=0.6.3",
        "fastapi[standard]>=0.110.0",
        "rapid_layout>=1.2.1",
        "rapidocr_onnxruntime>=1.4.0",
        "transformers>=4.45.0",
        "qwen-vl-utils>=0.0.8",
        "accelerate>=0.30.0",
        "torch>=2.4.0",
        "PyMuPDF>=1.24.0",
        "Pillow>=10.2.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
    )
    .env({
        "VLLM_USE_V1": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .run_commands(
        "python -c 'from rapid_layout import RapidLayout; RapidLayout(model_type=\"pp_doc_layoutv3\")'"
    )
)

PROMPTS = {
    "text": "OCR and recognize the text in this image crop. Maintain the original reading flow and formatting. Output only the recognized text without bounding boxes or coordinates:",
    "table": "Convert this table image crop into a clean, valid GitHub-flavored Markdown table. Preserve all columns, rows, headers, and numerical alignment accurately. Only output the markdown table:",
    "formula": "Convert this mathematical formula into clean LaTeX format. Wrap display formulas in $$ and inline formulas in $. Output only the LaTeX string without bounding boxes or coordinates:",
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
    Loads official pp_doc_layoutv3 model, segments document pages into
    semantic blocks (table, formula, text, figure), and extracts crops.
    """
    import pymupdf as fitz
    import numpy as np
    from rapid_layout import RapidLayout

    # Explicitly load official PP-DocLayoutV3 ONNX model
    try:
        layout_engine = RapidLayout(model_type="pp_doc_layoutv3", conf_thresh=0.25)
    except Exception as e:
        layout_engine = None
        print(f"Failed to initialize PP-DocLayoutV3: {e}")

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

        # Process each page through PP-DocLayoutV3
        for page_idx, page_img in enumerate(pil_images):
            img_np = np.array(page_img)
            boxes, scores, class_names = [], [], []

            if layout_engine is not None:
                try:
                    # rapid_layout 1.2+ returns a RapidLayoutOutput dataclass
                    res = layout_engine(img_np)
                    if res and res.boxes:
                        boxes = res.boxes
                        class_names = [str(c).lower() for c in res.class_names]
                        scores = res.scores
                except Exception as e:
                    print(f"Layout inference error on page {page_idx+1}: {e}")

            # Fallback if no layout boxes detected
            if not boxes:
                boxes = [[0, 0, page_img.width, page_img.height]]
                class_names = ["text"]
                scores = [1.0]

            # Reading order sorting: Top-to-bottom, left-to-right
            sorted_items = sorted(
                zip(boxes, class_names, scores),
                key=lambda item: (item[0][1] // 50, item[0][0])
            )

            # Crop semantic regions and serialize
            for b_idx, (box, raw_label, score) in enumerate(sorted_items):
                x0, y0, x1, y1 = [int(v) for v in box]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(page_img.width, x1), min(page_img.height, y1)

                if (x1 - x0) < 10 or (y1 - y0) < 10:
                    continue

                crop_pil = page_img.crop((x0, y0, x1, y1))
                buf = io.BytesIO()
                crop_pil.save(buf, format="JPEG", quality=95)
                crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                # Map PP-DocLayoutV3 labels to specialized categories
                cat = "text"
                if raw_label == "table":
                    cat = "table"
                elif "formula" in raw_label or "equation" in raw_label or raw_label == "algorithm":
                    cat = "formula"
                elif raw_label in ["chart", "image", "footer_image", "header_image", "figure"]:
                    cat = "figure"

                pages_crops.append({
                    "page": page_idx + 1,
                    "region_id": b_idx,
                    "raw_label": raw_label,
                    "label": cat,
                    "score": round(float(score), 3),
                    "bbox": [x0, y0, x1, y1],
                    "crop_b64": crop_b64,
                })

    except Exception as e:
        pages_crops.append({
            "page": 1,
            "region_id": 0,
            "raw_label": "error",
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
class VLLMTranscriber:
    """
    Stage 2: Targeted Multimodal Generative OCR served via vLLM engine.
    Uses continuous batching across document crops to maximize throughput on NVIDIA A10G.
    Model: Qwen/Qwen2-VL-2B-Instruct
    """
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"🚀 Initializing vLLM engine for {self.model_id} on NVIDIA A10G...")
        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.80,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
        )
        print(f"✅ vLLM engine ready for {self.model_id} on NVIDIA A10G.")

    @modal.method()
    def transcribe_crops(self, crops: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        markdown_sections = []

        if not crops:
            return {"markdown": "No crops detected.", "layout": {"regions": []}, "status": "done"}

        # 1. Prepare continuous batch inputs for vLLM
        vllm_inputs = []
        valid_crop_metas = []

        for crop_meta in crops:
            b64_crop = crop_meta.get("crop_b64", "")
            label = crop_meta.get("label", "text")
            if not b64_crop:
                continue

            crop_bytes = base64.b64decode(b64_crop)
            crop_pil = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
            prompt_text = PROMPTS.get(label, PROMPTS["text"])

            # Standard vLLM multimodal chat template with system header
            formatted_prompt = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            )
            vllm_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": crop_pil},
            })
            valid_crop_metas.append(crop_meta)

        if vllm_inputs:
            outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
            for crop_meta, output in zip(valid_crop_metas, outputs):
                text_out = output.outputs[0].text.strip()
                results.append({
                    "page": crop_meta.get("page", 1),
                    "region_id": crop_meta.get("region_id", 0),
                    "label": crop_meta.get("label", "text"),
                    "raw_label": crop_meta.get("raw_label", "text"),
                    "bbox": crop_meta.get("bbox", []),
                    "content": text_out,
                })
                markdown_sections.append(text_out)

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
    Unified HTTP Webhook for Gateway.
    Stage 1 (PP-DocLayoutV3 on T4) -> Stage 2 (vLLM Qwen2-VL-2B-Instruct on A10G) -> Structured Markdown.
    """
    raw_b64 = payload.get("data", "")
    extension = payload.get("extension", "pdf")
    file_bytes = base64.b64decode(raw_b64)

    # 1. Stage 1: PP-DocLayoutV3 Layout Detection on T4
    layout_result = parse_layout.remote(file_bytes, extension)
    regions = layout_result.get("regions", [])

    # 2. Stage 2: vLLM Continuous Batching Transcriber on A10G
    transcriber = VLLMTranscriber()
    ocr_result = transcriber.transcribe_crops.remote(regions)

    return {
        "task_id": payload.get("task_id", ""),
        "status": "done",
        "result": {
            "markdown": ocr_result.get("markdown", ""),
            "layout": ocr_result.get("layout", {}),
        },
    }


@app.local_entrypoint()
def main(file_path: str = "eval/dataset/images/02_financial_table.png"):
    """CLI test runner for end-to-end pipeline verification."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    ext = file_path.split(".")[-1]
    print(f"Testing document processing for {file_path} ({len(file_bytes)} bytes)...")

    # 1. Test layout detection
    layout_res = parse_layout.remote(file_bytes, ext)
    print(f"Stage 1 Complete: {len(layout_res['regions'])} regions detected.")
    for r in layout_res["regions"]:
        print(f"  - Region {r['region_id']}: {r['label']} (raw: {r['raw_label']}, score: {r['score']})")

    # 2. Test vLLM transcription
    transcriber = VLLMTranscriber()
    ocr_res = transcriber.transcribe_crops.remote(layout_res["regions"])
    print(f"\nStage 2 Complete (vLLM Output):\n{ocr_res['markdown']}")
