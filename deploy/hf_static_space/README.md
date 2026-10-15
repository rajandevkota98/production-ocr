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
- **Stage 2 (Generative SLM)**: Qwen 3.5 4B on NVIDIA A10G/A100 GPU (Modal)
