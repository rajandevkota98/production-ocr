---
title: Production OCR Platform
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: mit
---

# Production OCR Platform

Interactive live demo for Production-Grade Visual Document Understanding (VDU) and OCR.

### Architecture:
- **Frontend**: Hugging Face Static Space (Free CDN)
- **Stage 1 (Layout)**: PP-DocLayoutV3 on NVIDIA T4 GPU (Modal)
- **Stage 2 (Generative SLM)**: Qwen2-VL-2B-Instruct on NVIDIA A10G GPU via vLLM (Modal)
