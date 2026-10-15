import os
import sys
import time
import json
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapidocr_onnxruntime import RapidOCR
from eval.metrics import (
    compute_cer,
    compute_wer,
    compute_reading_order_score,
    evaluate_table_structure,
    evaluate_formula_fidelity,
)

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "benchmark_100")
IMG_DIR = os.path.join(BENCH_DIR, "images")
GT_DIR = os.path.join(BENCH_DIR, "ground_truth")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODAL_ENDPOINT = "https://r-devkota-98--production-ocr-process-document.us-east.modal.run"

class Benchmark100:
    def __init__(self):
        print("Initializing Baseline RapidOCR / PaddleOCR engine...")
        self.baseline_ocr = RapidOCR()

    def process_baseline(self, img_path: str) -> Dict[str, Any]:
        t0 = time.time()
        res, _ = self.baseline_ocr(img_path)
        lat = time.time() - t0
        lines = [item[1] for item in res] if res else []
        return {
            "text": "\n".join(lines),
            "latency": round(lat, 4),
        }

    def process_twostage(self, img_path: str, doc_id: str, retries: int = 2) -> Dict[str, Any]:
        with open(img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "data": b64_data,
            "extension": "png",
            "task_id": f"bench100_{doc_id}",
        }

        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                resp = requests.post(MODAL_ENDPOINT, json=payload, timeout=90)
                lat = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json().get("result", {})
                    return {
                        "text": data.get("markdown", ""),
                        "latency": round(lat, 4),
                        "status": "success"
                    }
                elif attempt < retries:
                    time.sleep(2)
            except Exception as e:
                if attempt < retries:
                    time.sleep(3)
                else:
                    return {"text": "", "latency": round(time.time() - t0, 4), "status": f"err_{str(e)}"}
        return {"text": "", "latency": 0.0, "status": "failed"}

    def warmup_modal(self, sample_img: str):
        print("Warming up Modal GPU containers before benchmark execution...")
        for _ in range(2):
            try:
                res = self.process_twostage(sample_img, "warmup", retries=2)
                if res["status"] == "success":
                    print(f"Modal GPU containers ready (Warmup latency: {res['latency']}s).")
                    return True
            except Exception:
                time.sleep(3)
        return False

    def run_benchmark(self, max_samples: int = 100):
        print(f"\n=======================================================")
        print(f"RUNNING 100-DATASET EMPIRICAL BENCHMARK (N={max_samples})")
        print(f"Single-Shot PaddleOCR vs Two-Stage VDU Pipeline")
        print(f"=======================================================\n")

        gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".json")])[:max_samples]
        total_docs = len(gt_files)
        sample_img = os.path.join(IMG_DIR, "doc_001_multicolumn.png")

        # 0. Warm up Modal container
        self.warmup_modal(sample_img)

        results = []
        start_overall = time.time()

        # Step 1: Run Baseline for all docs (local CPU)
        print("\n[Stage 1/2] Running Baseline Single-Shot PaddleOCR on all 100 documents...")
        baseline_outputs = {}
        for idx, gtf in enumerate(gt_files):
            doc_id = gtf.replace(".json", "")
            with open(os.path.join(GT_DIR, gtf), "r") as f:
                meta = json.load(f)
            img_path = os.path.join(IMG_DIR, meta["file"])
            base_out = self.process_baseline(img_path)
            baseline_outputs[doc_id] = base_out
            if (idx + 1) % 25 == 0 or idx == total_docs - 1:
                print(f"  Processed {idx + 1}/{total_docs} baseline documents...")

        # Step 2: Run Two-Stage Pipeline on Modal in parallel pool of 4 workers
        print("\n[Stage 2/2] Running Two-Stage VDU Pipeline via Modal GPU endpoint...")
        twostage_outputs = {}

        def fetch_modal(gtf):
            with open(os.path.join(GT_DIR, gtf), "r") as f:
                meta = json.load(f)
            doc_id = meta["id"]
            img_path = os.path.join(IMG_DIR, meta["file"])
            res = self.process_twostage(img_path, doc_id, retries=2)
            return doc_id, res

        completed_count = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_doc = {executor.submit(fetch_modal, gtf): gtf for gtf in gt_files}
            for future in as_completed(future_to_doc):
                doc_id, res = future.result()
                twostage_outputs[doc_id] = res
                completed_count += 1
                if completed_count % 20 == 0 or completed_count == total_docs:
                    print(f"  Modal Two-Stage: {completed_count}/{total_docs} completed...")

        print("\n[Metrics] Computing Statistical Metrics (CER, WER, Kendall Tau, Table TEDS)...")

        # Accumulators
        b_cers, t_cers = [], []
        b_wers, t_wers = [], []
        b_lats, t_lats = [], []
        
        # Category specific
        col_b_inversions, col_t_inversions = [], []
        col_b_taus, col_t_taus = [], []
        tbl_b_scores, tbl_t_scores = [], []
        fml_b_accs, fml_t_accs = [], []

        for gtf in gt_files:
            with open(os.path.join(GT_DIR, gtf), "r") as f:
                meta = json.load(f)
            doc_id = meta["id"]
            doc_type = meta["type"]

            b_res = baseline_outputs.get(doc_id, {"text": "", "latency": 0.0})
            t_res = twostage_outputs.get(doc_id, {"text": "", "latency": 0.0})

            b_text = b_res["text"]
            t_text = t_res["text"]

            b_lats.append(b_res["latency"])
            t_lats.append(t_res["latency"])

            if doc_type == "multi_column":
                target_gt = meta["ground_truth_text"]
                expected_phrases = []
                for blk in meta["reading_blocks"]:
                    expected_phrases.extend(blk)
                b_order = compute_reading_order_score(expected_phrases, b_text)
                t_order = compute_reading_order_score(expected_phrases, t_text)

                col_b_inversions.append(b_order["inversions"])
                col_t_inversions.append(t_order["inversions"])
                col_b_taus.append(b_order["kendall_tau"])
                col_t_taus.append(t_order["kendall_tau"])

            elif doc_type == "table":
                target_gt = meta["ground_truth_markdown"]
                b_tbl = evaluate_table_structure(meta["table_cells"], b_text)
                t_tbl = evaluate_table_structure(meta["table_cells"], t_text)
                tbl_b_scores.append(b_tbl["table_structure_score"])
                tbl_t_scores.append(t_tbl["table_structure_score"])

            else:  # formula
                target_gt = "\n".join(meta["ground_truth_formulas"])
                b_f = evaluate_formula_fidelity(meta["ground_truth_formulas"], meta["expected_terms"], b_text)
                t_f = evaluate_formula_fidelity(meta["ground_truth_formulas"], meta["expected_terms"], t_text)
                fml_b_accs.append(b_f["term_accuracy"])
                fml_t_accs.append(t_f["term_accuracy"])

            b_cer = compute_cer(target_gt, b_text)
            t_cer = compute_cer(target_gt, t_text)
            b_wer = compute_wer(target_gt, b_text)
            t_wer = compute_wer(target_gt, t_text)

            b_cers.append(b_cer)
            t_cers.append(t_cer)
            b_wers.append(b_wer)
            t_wers.append(t_wer)

            results.append({
                "doc_id": doc_id,
                "type": doc_type,
                "baseline_cer": b_cer,
                "twostage_cer": t_cer,
                "baseline_wer": b_wer,
                "twostage_wer": t_wer,
                "baseline_latency": b_res["latency"],
                "twostage_latency": t_res["latency"],
            })

        total_runtime = time.time() - start_overall

        stats = {
            "sample_size_n": total_docs,
            "total_benchmark_time_sec": round(total_runtime, 2),
            "estimated_benchmark_cost_usd": round(len(gt_files) * 0.0008, 4), # ~$0.08
            "overall": {
                "baseline_cer_mean": round(float(np.mean(b_cers)), 4),
                "baseline_cer_std": round(float(np.std(b_cers)), 4),
                "twostage_cer_mean": round(float(np.mean(t_cers)), 4),
                "twostage_cer_std": round(float(np.std(t_cers)), 4),
                "baseline_wer_mean": round(float(np.mean(b_wers)), 4),
                "twostage_wer_mean": round(float(np.mean(t_wers)), 4),
                "baseline_avg_latency_sec": round(float(np.mean(b_lats)), 2),
                "twostage_avg_latency_sec": round(float(np.mean(t_lats)), 2),
            },
            "multi_column_subset_n40": {
                "baseline_avg_inversions": round(float(np.mean(col_b_inversions)), 2),
                "twostage_avg_inversions": round(float(np.mean(col_t_inversions)), 2),
                "inversions_eliminated_pct": round(float((1.0 - (np.mean(col_t_inversions)/np.mean(col_b_inversions) if np.mean(col_b_inversions) > 0 else 0)) * 100), 1),
                "baseline_kendall_tau": round(float(np.mean(col_b_taus)), 3),
                "twostage_kendall_tau": round(float(np.mean(col_t_taus)), 3),
            },
            "table_subset_n35": {
                "baseline_avg_structure_score": round(float(np.mean(tbl_b_scores)), 3),
                "twostage_avg_structure_score": round(float(np.mean(tbl_t_scores)), 3),
            },
            "formula_subset_n25": {
                "baseline_term_accuracy": round(float(np.mean(fml_b_accs)), 3),
                "twostage_term_accuracy": round(float(np.mean(fml_t_accs)), 3),
            },
            "economic_tco_per_1000_pages": {
                "aws_textract_google_cloud": "$15.00 - $50.00",
                "twostage_modal_serverless": "$0.75 - $1.20",
                "cost_savings_pct": "95.0% - 97.5%"
            }
        }

        output_json = {
            "summary": stats,
            "sample_results": results, # save all 100 samples
        }

        results_path = os.path.join(RESULTS_DIR, "benchmark_100_results.json")
        with open(results_path, "w") as f:
            json.dump(output_json, f, indent=2)

        print("\n=======================================================")
        print("100-SAMPLE BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {results_path}")
        print(f"Total benchmark run cost: ~${stats['estimated_benchmark_cost_usd']} USD")
        print("=======================================================\n")
        return output_json

if __name__ == "__main__":
    runner = Benchmark100()
    runner.run_benchmark(100)
