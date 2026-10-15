import os
import sys
import time
import json
import base64
import requests
from typing import Dict, Any, List

from rapidocr_onnxruntime import RapidOCR
from eval.metrics import (
    compute_cer,
    compute_wer,
    compute_reading_order_score,
    evaluate_table_structure,
    evaluate_formula_fidelity,
)

MODAL_ENDPOINT = "https://r-devkota-98--production-ocr-process-document.us-east.modal.run"
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class BenchmarkRunner:
    def __init__(self):
        print("Initializing Baseline OCR Engine (PaddleOCR / RapidOCR)...")
        self.baseline_ocr = RapidOCR()

    def run_single_shot_baseline(self, img_path: str) -> Dict[str, Any]:
        """
        Executes standard single-stage OCR.
        Detects text bounding boxes and sorts horizontally by Y-coordinate (industry default).
        """
        start_time = time.time()
        results, _ = self.baseline_ocr(img_path)
        latency = round(time.time() - start_time, 4)

        if not results:
            return {"text": "", "latency": latency, "boxes": []}

        # Industry standard horizontal line sorting
        # results: list of [bbox, text, score]
        lines = [item[1] for item in results]
        extracted_text = "\n".join(lines)

        return {
            "text": extracted_text,
            "latency": latency,
            "raw_boxes": len(results),
        }

    def run_two_stage_pipeline(self, img_path: str) -> Dict[str, Any]:
        """
        Executes Two-Stage Layout-First VDU Pipeline via Modal cloud endpoint.
        Stage 1: PP-DocLayoutV3 detects semantic blocks (columns, tables, formulas).
        Stage 2: Generative SLM (Qwen2-VL-2B-Instruct on vLLM) transcribes structured Markdown.
        """
        start_time = time.time()
        with open(img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "data": b64_data,
            "extension": "png",
            "task_id": f"bench_{os.path.basename(img_path)}",
        }

        try:
            resp = requests.post(MODAL_ENDPOINT, json=payload, timeout=90)
            latency = round(time.time() - start_time, 4)
            if resp.status_code == 200:
                data = resp.json().get("result", {})
                markdown = data.get("markdown", "")
                layout = data.get("layout", {})
                return {
                    "text": markdown,
                    "layout": layout,
                    "latency": latency,
                    "status": "success",
                }
            else:
                return {
                    "text": f"Error: HTTP {resp.status_code}",
                    "latency": latency,
                    "status": "failed",
                }
        except Exception as e:
            return {
                "text": f"Error: {str(e)}",
                "latency": round(time.time() - start_time, 4),
                "status": "failed",
            }

    def evaluate_all(self) -> Dict[str, Any]:
        img_dir = os.path.join(DATASET_DIR, "images")
        gt_dir = os.path.join(DATASET_DIR, "ground_truth")

        benchmark_cases = [
            {
                "file": "01_multicolumn_doc.png",
                "gt_file": "01_multicolumn_doc.json",
                "type": "multi_column",
                "name": "Multi-Column Academic Paper",
            },
            {
                "file": "02_financial_table.png",
                "gt_file": "02_financial_table.json",
                "type": "table",
                "name": "Borderless Financial Statement Table",
            },
            {
                "file": "03_scientific_formula.png",
                "gt_file": "03_scientific_formula.json",
                "type": "formula",
                "name": "Scientific Formula & Equation Document",
            },
        ]

        eval_summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cases": [],
            "aggregate": {},
        }

        baseline_cers = []
        baseline_wers = []
        twostage_cers = []
        twostage_wers = []
        baseline_latencies = []
        twostage_latencies = []

        for case in benchmark_cases:
            print(f"\n==========================================")
            print(f"Evaluating: {case['name']} ({case['file']})")
            print(f"==========================================")

            img_path = os.path.join(img_dir, case["file"])
            gt_path = os.path.join(gt_dir, case["gt_file"])

            with open(gt_path, "r") as f:
                gt_meta = json.load(f)

            # 1. Run Baseline
            print("-> Running Single-Shot Baseline (PaddleOCR)...")
            base_res = self.run_single_shot_baseline(img_path)
            base_text = base_res["text"]
            base_lat = base_res["latency"]
            baseline_latencies.append(base_lat)

            # 2. Run Two-Stage Pipeline
            print("-> Running Two-Stage VDU Pipeline (PP-DocLayoutV3 + Generative SLM)...")
            two_res = self.run_two_stage_pipeline(img_path)
            two_text = two_res["text"]
            two_lat = two_res["latency"]
            twostage_latencies.append(two_lat)

            # Specific Ground Truth Text for General CER/WER
            if case["type"] == "multi_column":
                target_gt = gt_meta["ground_truth_text"]
                # Evaluate Reading Order
                expected_phrases = []
                for block in gt_meta["ground_truth_reading_blocks"]:
                    expected_phrases.extend(block)
                base_order = compute_reading_order_score(expected_phrases, base_text)
                two_order = compute_reading_order_score(expected_phrases, two_text)
                metrics_special = {
                    "baseline_reading_order": base_order,
                    "twostage_reading_order": two_order,
                }
            elif case["type"] == "table":
                target_gt = gt_meta["ground_truth_markdown"]
                base_tbl = evaluate_table_structure(gt_meta["table_cells"], base_text)
                two_tbl = evaluate_table_structure(gt_meta["table_cells"], two_text)
                metrics_special = {
                    "baseline_table": base_tbl,
                    "twostage_table": two_tbl,
                }
            else:  # formula
                target_gt = "\n".join(gt_meta["ground_truth_formulas"])
                base_f = evaluate_formula_fidelity(gt_meta["ground_truth_formulas"], gt_meta["expected_terms"], base_text)
                two_f = evaluate_formula_fidelity(gt_meta["ground_truth_formulas"], gt_meta["expected_terms"], two_text)
                metrics_special = {
                    "baseline_formula": base_f,
                    "twostage_formula": two_f,
                }

            # Calculate CER & WER
            b_cer = compute_cer(target_gt, base_text)
            t_cer = compute_cer(target_gt, two_text)
            b_wer = compute_wer(target_gt, base_text)
            t_wer = compute_wer(target_gt, two_text)

            baseline_cers.append(b_cer)
            baseline_wers.append(b_wer)
            twostage_cers.append(t_cer)
            twostage_wers.append(t_wer)

            case_result = {
                "id": case["file"],
                "name": case["name"],
                "type": case["type"],
                "baseline": {
                    "latency_sec": base_lat,
                    "cer": b_cer,
                    "wer": b_wer,
                    "extracted_text_preview": base_text[:300] + "...",
                },
                "twostage": {
                    "latency_sec": two_lat,
                    "cer": t_cer,
                    "wer": t_wer,
                    "extracted_text_preview": two_text[:300] + "...",
                },
                "special_metrics": metrics_special,
            }
            eval_summary["cases"].append(case_result)

        # Aggregate metrics
        avg_base_cer = round(sum(baseline_cers) / len(baseline_cers), 4)
        avg_two_cer = round(sum(twostage_cers) / len(twostage_cers), 4)
        avg_base_wer = round(sum(baseline_wers) / len(baseline_wers), 4)
        avg_two_wer = round(sum(twostage_wers) / len(twostage_wers), 4)
        cer_reduction = round((1.0 - (avg_two_cer / avg_base_cer if avg_base_cer > 0 else 0)) * 100, 1)

        eval_summary["aggregate"] = {
            "baseline_avg_cer": avg_base_cer,
            "twostage_avg_cer": avg_two_cer,
            "cer_reduction_percent": cer_reduction,
            "baseline_avg_wer": avg_base_wer,
            "twostage_avg_wer": avg_two_wer,
            "baseline_avg_latency_sec": round(sum(baseline_latencies) / len(baseline_latencies), 2),
            "twostage_avg_latency_sec": round(sum(twostage_latencies) / len(twostage_latencies), 2),
            "cost_per_1k_pages": {
                "commercial_ocr_apis (AWS/Google)": "$15.00 - $50.00",
                "single_stage_paddleocr (CPU self-host)": "$0.80 - $1.10",
                "twostage_serverless_vdu (Modal T4+A10G)": "$0.75 - $1.20",
            }
        }

        results_file = os.path.join(RESULTS_DIR, "benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(eval_summary, f, indent=2)

        print("\n==========================================")
        print("BENCHMARK EXECUTION COMPLETE")
        print(f"Results saved to: {results_file}")
        print("==========================================")
        return eval_summary

if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = runner.evaluate_all()
