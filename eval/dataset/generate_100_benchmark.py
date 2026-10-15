import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(BASE_DIR, "benchmark_100")
IMG_DIR = os.path.join(BENCH_DIR, "images")
GT_DIR = os.path.join(BENCH_DIR, "ground_truth")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

FONT_TITLE = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
FONT_HEAD = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18)
FONT_BODY = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 15)
FONT_MONO = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15)

TOPICS_COL = [
    ("Distributed Systems", "Cloud scalability requires autonomous microservices architecture.", "Failure recovery protocols must execute without manual intervention.", "Consensus algorithms provide atomic state across nodes.", "Network partitions trigger fallback election mechanisms."),
    ("Machine Learning Operations", "Continuous model training pipelines validate data drift.", "Feature store caching reduces latency for real-time inference.", "Automated model registry tracks model weights and lineage.", "Drift detection alerts trigger automated retrain workflows."),
    ("Financial Risk Modeling", "Portfolio value-at-risk calculates maximum expected losses.", "Monte Carlo simulations stress test extreme market events.", "Liquidity risk frameworks monitor capital adequacy ratios.", "Credit default swap spreads indicate counterparty solvency."),
    ("Healthcare Informatics", "Electronic health record standards ensure patient data privacy.", "Federated learning enables cross-hospital diagnostic models.", "Clinical NLP extracts structured terms from physician notes.", "Audit trails ensure strict compliance with HIPAA mandates."),
    ("Enterprise Cybersecurity", "Zero-trust network architecture enforces identity verification.", "Microsegmentation confines lateral movement during breach.", "Automated patch management reduces vulnerable software surfaces.", "Encrypted transport layers protect sensitive payload transit."),
]

def generate_multicolumn_sample(idx: int):
    topic, l1, l2, r1, r2 = TOPICS_COL[idx % len(TOPICS_COL)]
    width, height = 1100, 700
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title = f"Document #{idx+1:03d}: Technical Analysis of {topic}"
    draw.text((60, 40), title, fill=(20, 20, 20), font=FONT_TITLE)
    draw.line([(60, 80), (1040, 80)], fill=(210, 210, 210), width=2)

    col1 = [
        f"1. Primary Architecture Overview",
        l1,
        l2,
        f"Key benchmark targets high throughput under load.",
        f"System latency remains sub-50ms at peak traffic.",
    ]
    col2 = [
        f"2. Implementation & Empirical Results",
        r1,
        r2,
        f"Decoupled components scale horizontally on demand.",
        f"Empirical verification confirms zero data loss.",
    ]

    # Draw Col 1
    y = 110
    draw.text((60, y), col1[0], fill=(20, 20, 20), font=FONT_HEAD)
    y += 35
    for text in col1[1:]:
        draw.text((60, y), text, fill=(50, 50, 50), font=FONT_BODY)
        y += 38

    # Draw Col 2
    y = 110
    draw.text((560, y), col2[0], fill=(20, 20, 20), font=FONT_HEAD)
    y += 35
    for text in col2[1:]:
        draw.text((560, y), text, fill=(50, 50, 50), font=FONT_BODY)
        y += 38

    file_name = f"doc_{idx+1:03d}_multicolumn.png"
    img.save(os.path.join(IMG_DIR, file_name), dpi=(300, 300))

    gt_text = f"{title}\n\n" + "\n".join(col1) + "\n\n" + "\n".join(col2)
    meta = {
        "id": f"doc_{idx+1:03d}",
        "type": "multi_column",
        "file": file_name,
        "ground_truth_text": gt_text,
        "reading_blocks": [col1, col2]
    }
    with open(os.path.join(GT_DIR, f"doc_{idx+1:03d}.json"), "w") as f:
        json.dump(meta, f, indent=2)

TABLE_METRICS = [
    ["Cloud Infrastructure", "$140.2", "$185.6", "+32.4%"],
    ["Enterprise AI SaaS", "$92.1", "$135.4", "+47.0%"],
    ["API Gateway Tier", "$45.0", "$56.2", "+24.9%"],
    ["Storage Engine", "$38.5", "$42.1", "+9.4%"],
    ["Professional Support", "$28.0", "$31.5", "+12.5%"],
    ["Security Shield", "$19.4", "$26.8", "+38.1%"],
]

def generate_table_sample(idx: int):
    width, height = 1050, 550
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title = f"Document #{idx+1:03d}: Financial Statement (Segment Breakdown)"
    draw.text((50, 35), title, fill=(20, 20, 20), font=FONT_TITLE)
    draw.text((50, 75), "Unaudited quarterly revenue reported in Millions USD", fill=(100, 100, 100), font=FONT_BODY)

    headers = ["Business Segment", "Fiscal Q1 ($M)", "Fiscal Q2 ($M)", "YoY Growth"]
    cols = [50, 380, 580, 780]
    sub_rows = random.sample(TABLE_METRICS, 4)

    y = 120
    draw.rectangle([(40, y - 5), (1010, y + 35)], fill=(240, 244, 248))
    for c_idx, h in enumerate(headers):
        draw.text((cols[c_idx], y + 5), h, fill=(20, 20, 20), font=FONT_HEAD)
    draw.line([(40, y + 35), (1010, y + 35)], fill=(180, 180, 180), width=2)

    y = 165
    for r_idx, row in enumerate(sub_rows):
        if r_idx % 2 == 1:
            draw.rectangle([(40, y - 5), (1010, y + 35)], fill=(250, 250, 252))
        for c_idx, val in enumerate(row):
            draw.text((cols[c_idx], y + 5), val, fill=(40, 40, 40), font=FONT_BODY)
        draw.line([(40, y + 35), (1010, y + 35)], fill=(225, 225, 225), width=1)
        y += 45

    file_name = f"doc_{idx+1:03d}_table.png"
    img.save(os.path.join(IMG_DIR, file_name), dpi=(300, 300))

    md_table = f"| {' | '.join(headers)} |\n|---|---|---|---|\n"
    for row in sub_rows:
        md_table += f"| {' | '.join(row)} |\n"

    meta = {
        "id": f"doc_{idx+1:03d}",
        "type": "table",
        "file": file_name,
        "ground_truth_markdown": md_table.strip(),
        "table_cells": [headers] + sub_rows
    }
    with open(os.path.join(GT_DIR, f"doc_{idx+1:03d}.json"), "w") as f:
        json.dump(meta, f, indent=2)

FORMULAS = [
    ("Loss Function", "L(w) = (1/N) * sum_{i=1}^{N} (y_i - f(x_i; w))^2 + lambda * ||w||^2", ["L(w)", "sum", "lambda"]),
    ("Softmax Cross-Entropy", "P(y=k | x) = exp(z_k) / sum_{j=1}^{C} exp(z_j)", ["Softmax", "exp", "sum"]),
    ("Gaussian Probability", "p(x | mu, sigma) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))", ["sigma", "sqrt", "pi", "exp", "mu"]),
    ("Information Entropy", "H(X) = - sum_{i=1}^{n} P(x_i) * log2(P(x_i))", ["H(X)", "sum", "log2"]),
]

def generate_formula_sample(idx: int):
    width, height = 1000, 500
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title, f_eq, terms = FORMULAS[idx % len(FORMULAS)]
    doc_title = f"Document #{idx+1:03d}: Mathematical Definition - {f_title}"
    draw.text((50, 40), doc_title, fill=(20, 20, 20), font=FONT_TITLE)
    draw.text((50, 90), f"Standard analytical formulation for {f_title.lower()}:", fill=(60, 60, 60), font=FONT_BODY)

    draw.rectangle([(50, 140), (950, 240)], fill=(245, 248, 252), outline=(210, 220, 235), width=1)
    draw.text((80, 180), f_eq, fill=(15, 30, 80), font=FONT_MONO)

    draw.text((50, 280), "Where parameters represent normalized distributions over empirical observations.", fill=(80, 80, 80), font=FONT_BODY)

    file_name = f"doc_{idx+1:03d}_formula.png"
    img.save(os.path.join(IMG_DIR, file_name), dpi=(300, 300))

    meta = {
        "id": f"doc_{idx+1:03d}",
        "type": "formula",
        "file": file_name,
        "ground_truth_formulas": [f_eq],
        "expected_terms": terms
    }
    with open(os.path.join(GT_DIR, f"doc_{idx+1:03d}.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main():
    print("Generating exactly 100 benchmark documents across 3 industry categories...")
    # 40 Multi-Column (1 - 40)
    for i in range(40):
        generate_multicolumn_sample(i)
    # 35 Tables (40 - 75)
    for i in range(40, 75):
        generate_table_sample(i)
    # 25 Formulas (75 - 100)
    for i in range(75, 100):
        generate_formula_sample(i)

    print("Successfully generated 100 benchmark documents with ground-truth metadata in eval/dataset/benchmark_100/!")

if __name__ == "__main__":
    main()
