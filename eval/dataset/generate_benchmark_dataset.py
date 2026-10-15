import os
import json
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(DATASET_DIR, "images")
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

# Select clean system font
FONT_TITLE = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 26)
FONT_HEADING = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
FONT_BODY = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
FONT_MONO = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 16)

def generate_multicolumn():
    """Generates a two-column academic document testing reading order interleaving."""
    width, height = 1200, 800
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    title = "Empirical Evaluation of Document Understanding Architectures"
    draw.text((80, 50), title, fill=(20, 20, 20), font=FONT_TITLE)
    draw.line([(80, 90), (1120, 90)], fill=(200, 200, 200), width=2)

    col1_lines = [
        ("1. Introduction & Motivation", FONT_HEADING),
        ("Deep visual document understanding requires robust layout awareness.", FONT_BODY),
        ("Traditional single-shot OCR systems scan text lines strictly horizontally.", FONT_BODY),
        ("On multi-column layouts, horizontal scans interleave distinct columns.", FONT_BODY),
        ("This results in fragmented sentences and jumbled semantic meaning.", FONT_BODY),
        ("Enterprise search and LLM retrieval pipelines suffer catastrophic failure.", FONT_BODY),
        ("Therefore, layout-first segmentation is mandatory for multi-column text.", FONT_BODY),
    ]

    col2_lines = [
        ("2. Two-Stage Vision Pipeline", FONT_HEADING),
        ("Our two-stage approach decouples layout segmentation from OCR.", FONT_BODY),
        ("Stage 1 utilizes PP-DocLayoutV3 to detect column bounding boxes.", FONT_BODY),
        ("Stage 2 dispatches cropped semantic regions to a specialized SLM.", FONT_BODY),
        ("Text is parsed sequentially down column 1 before traversing column 2.", FONT_BODY),
        ("This preserves natural human reading order with 100% structural fidelity.", FONT_BODY),
        ("Extracted markdown cleanly mirrors original document intent and syntax.", FONT_BODY),
    ]

    # Draw Column 1
    y = 120
    for text, font in col1_lines:
        draw.text((80, y), text, fill=(30, 30, 30), font=font)
        y += 42

    # Draw Column 2
    y = 120
    for text, font in col2_lines:
        draw.text((620, y), text, fill=(30, 30, 30), font=font)
        y += 42

    img_path = os.path.join(IMG_DIR, "01_multicolumn_doc.png")
    img.save(img_path, dpi=(300, 300))

    # Ground truth: strict sequential column reading order
    gt_text = (
        "Empirical Evaluation of Document Understanding Architectures\n\n"
        "1. Introduction & Motivation\n"
        "Deep visual document understanding requires robust layout awareness.\n"
        "Traditional single-shot OCR systems scan text lines strictly horizontally.\n"
        "On multi-column layouts, horizontal scans interleave distinct columns.\n"
        "This results in fragmented sentences and jumbled semantic meaning.\n"
        "Enterprise search and LLM retrieval pipelines suffer catastrophic failure.\n"
        "Therefore, layout-first segmentation is mandatory for multi-column text.\n\n"
        "2. Two-Stage Vision Pipeline\n"
        "Our two-stage approach decouples layout segmentation from OCR.\n"
        "Stage 1 utilizes PP-DocLayoutV3 to detect column bounding boxes.\n"
        "Stage 2 dispatches cropped semantic regions to a specialized SLM.\n"
        "Text is parsed sequentially down column 1 before traversing column 2.\n"
        "This preserves natural human reading order with 100% structural fidelity.\n"
        "Extracted markdown cleanly mirrors original document intent and syntax."
    )

    gt_data = {
        "id": "01_multicolumn_doc",
        "category": "multi_column",
        "description": "Two-column technical paper testing reading-order preservation vs horizontal line interleaving",
        "ground_truth_text": gt_text,
        "ground_truth_reading_blocks": [
            [t[0] for t in col1_lines],
            [t[0] for t in col2_lines],
        ]
    }
    with open(os.path.join(GT_DIR, "01_multicolumn_doc.json"), "w") as f:
        json.dump(gt_data, f, indent=2)


def generate_table():
    """Generates a complex borderless financial statement table."""
    width, height = 1100, 600
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Document Header
    draw.text((60, 40), "Q2 Consolidated Financial Summary", fill=(20, 20, 20), font=FONT_TITLE)
    draw.text((60, 80), "Financial metrics reported in Millions USD (unaudited)", fill=(90, 90, 90), font=FONT_BODY)

    # Table coordinates
    cols = [60, 380, 580, 780, 960]
    headers = ["Financial Metric", "Q1 2024 ($M)", "Q2 2024 ($M)", "YoY Growth (%)"]
    rows = [
        ["Cloud Infrastructure", "$142.5", "$189.2", "+32.8%"],
        ["Enterprise AI Licensing", "$88.4", "$124.1", "+40.4%"],
        ["Professional Services", "$34.1", "$39.5", "+15.8%"],
        ["Operating Expenses", "$64.2", "$71.5", "+11.3%"],
        ["Net Operating Income", "$200.8", "$281.3", "+40.1%"],
    ]

    # Draw table header
    y = 140
    draw.rectangle([(50, y - 5), (1050, y + 35)], fill=(240, 243, 246))
    for i, h in enumerate(headers):
        draw.text((cols[i], y + 5), h, fill=(10, 10, 10), font=FONT_HEADING)
    draw.line([(50, y + 35), (1050, y + 35)], fill=(160, 160, 160), width=2)

    # Draw table rows (borderless, alternating subtle background)
    y = 185
    for r_idx, row in enumerate(rows):
        if r_idx % 2 == 1:
            draw.rectangle([(50, y - 5), (1050, y + 35)], fill=(250, 250, 252))
        for c_idx, val in enumerate(row):
            draw.text((cols[c_idx], y + 5), val, fill=(40, 40, 40), font=FONT_BODY)
        draw.line([(50, y + 35), (1050, y + 35)], fill=(230, 230, 230), width=1)
        y += 45

    img_path = os.path.join(IMG_DIR, "02_financial_table.png")
    img.save(img_path, dpi=(300, 300))

    # Ground truth Markdown table
    gt_markdown_table = (
        "| Financial Metric | Q1 2024 ($M) | Q2 2024 ($M) | YoY Growth (%) |\n"
        "|---|---|---|---|\n"
        "| Cloud Infrastructure | $142.5 | $189.2 | +32.8% |\n"
        "| Enterprise AI Licensing | $88.4 | $124.1 | +40.4% |\n"
        "| Professional Services | $34.1 | $39.5 | +15.8% |\n"
        "| Operating Expenses | $64.2 | $71.5 | +11.3% |\n"
        "| Net Operating Income | $200.8 | $281.3 | +40.1% |"
    )

    gt_data = {
        "id": "02_financial_table",
        "category": "table",
        "description": "Borderless financial table testing cell alignment, column association, and markdown structure",
        "ground_truth_markdown": gt_markdown_table,
        "table_cells": [headers] + rows
    }
    with open(os.path.join(GT_DIR, "02_financial_table.json"), "w") as f:
        json.dump(gt_data, f, indent=2)


def generate_formula():
    """Generates technical document containing mathematical equations and LaTeX symbols."""
    width, height = 1100, 550
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((60, 40), "Statistical Loss Formulation", fill=(20, 20, 20), font=FONT_TITLE)
    draw.text((60, 90), "1. Normalized Objective Function", fill=(30, 30, 30), font=FONT_HEADING)
    draw.text((60, 130), "The regularized empirical risk minimization problem is formulated as follows:", fill=(50, 50, 50), font=FONT_BODY)

    # Formula box
    draw.rectangle([(60, 175), (1040, 260)], fill=(245, 247, 250), outline=(210, 215, 225), width=1)
    formula_text_1 = "L(theta) = (1 / N) * sum_{i=1}^{N} L_ce(f(x_i; theta), y_i) + lambda * ||theta||_2^2"
    draw.text((90, 205), formula_text_1, fill=(10, 30, 80), font=FONT_MONO)

    draw.text((60, 285), "2. Gaussian Probability Density", fill=(30, 30, 30), font=FONT_HEADING)
    draw.rectangle([(60, 330), (1040, 415)], fill=(245, 247, 250), outline=(210, 215, 225), width=1)
    formula_text_2 = "p(x | mu, sigma^2) = (1 / (sigma * sqrt(2 * pi))) * exp( - (x - mu)^2 / (2 * sigma^2) )"
    draw.text((90, 360), formula_text_2, fill=(10, 30, 80), font=FONT_MONO)

    img_path = os.path.join(IMG_DIR, "03_scientific_formula.png")
    img.save(img_path, dpi=(300, 300))

    gt_data = {
        "id": "03_scientific_formula",
        "category": "formula",
        "description": "Mathematical formula document testing LaTeX syntax fidelity vs garbled ASCII text",
        "ground_truth_formulas": [
            r"L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{ce}(f(x_i; \theta), y_i) + \lambda \|\theta\|_2^2",
            r"p(x \mid \mu, \sigma^2) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( - \frac{(x - \mu)^2}{2\sigma^2} \right)"
        ],
        "expected_terms": [
            "L(theta)", "sum", "lambda", "exp", "sqrt", "sigma", "pi", "mu"
        ]
    }
    with open(os.path.join(GT_DIR, "03_scientific_formula.json"), "w") as f:
        json.dump(gt_data, f, indent=2)

if __name__ == "__main__":
    generate_multicolumn()
    generate_table()
    generate_formula()
    print("Benchmark test images and ground truth annotations generated successfully in eval/dataset/.")
