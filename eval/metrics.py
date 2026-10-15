import re
from typing import List, Tuple, Dict, Any
from rapidfuzz.distance import Levenshtein

def compute_cer(gt_text: str, pred_text: str) -> float:
    """
    Computes Character Error Rate (CER) via normalized Levenshtein distance.
    CER = Edit_Distance(gt, pred) / len(gt)
    """
    gt_clean = re.sub(r"\s+", " ", gt_text).strip()
    pred_clean = re.sub(r"\s+", " ", pred_text).strip()
    if not gt_clean:
        return 0.0 if not pred_clean else 1.0
    dist = Levenshtein.distance(gt_clean, pred_clean)
    return round(float(dist) / len(gt_clean), 4)

def compute_wer(gt_text: str, pred_text: str) -> float:
    """
    Computes Word Error Rate (WER).
    WER = Word_Edit_Distance(gt_words, pred_words) / len(gt_words)
    """
    gt_words = re.findall(r"\b\w+\b", gt_text)
    pred_words = re.findall(r"\b\w+\b", pred_text)
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    dist = Levenshtein.distance(gt_words, pred_words)
    return round(float(dist) / len(gt_words), 4)

def compute_reading_order_score(expected_phrases: List[str], extracted_text: str) -> Dict[str, Any]:
    """
    Evaluates multi-column reading order integrity.
    Computes Kendall's Tau and Inversion Count between ground-truth phrase sequence
    and the order they appear in the extracted text.
    """
    positions = []
    found_phrases = []

    for phrase in expected_phrases:
        # Search for case-insensitive match of the first 4-5 words
        snippet = " ".join(phrase.split()[:4]).lower()
        idx = extracted_text.lower().find(snippet)
        if idx != -1:
            positions.append(idx)
            found_phrases.append(phrase)

    n = len(positions)
    if n < 2:
        return {"kendall_tau": 0.0, "inversions": 0, "pairs": 0, "reading_order_accuracy": 0.0}

    # Count concordant vs discordant pairs
    concordant = 0
    discordant = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            if positions[i] < positions[j]:
                concordant += 1
            elif positions[i] > positions[j]:
                discordant += 1

    tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
    # Reading order accuracy percentage: 100% when 0 inversions
    order_acc = max(0.0, (concordant / total_pairs) * 100) if total_pairs > 0 else 0.0

    return {
        "kendall_tau": round(tau, 3),
        "inversions": discordant,
        "total_pairs": total_pairs,
        "reading_order_accuracy": round(order_acc, 1)
    }

def evaluate_table_structure(gt_cells: List[List[str]], pred_text: str) -> Dict[str, Any]:
    """
    Evaluates whether tabular information was extracted with correct row-column structure
    (e.g., Markdown table format) vs flat unstructured text dump.
    """
    is_markdown_table = False
    lines = [line.strip() for line in pred_text.splitlines() if line.strip()]
    pipe_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
    if len(pipe_lines) >= 3:
        is_markdown_table = True

    # Search for each cell in predicted output
    total_cells = sum(len(row) for row in gt_cells)
    cells_found = 0

    for row in gt_cells:
        for cell in row:
            clean_cell = cell.strip("$+ %").lower()
            if clean_cell and clean_cell in pred_text.lower():
                cells_found += 1

    cell_recall = round(cells_found / total_cells, 3) if total_cells > 0 else 0.0

    # Structural Score (TEDS-inspired):
    # If returned as proper Markdown table, give structure bonus + cell fidelity
    if is_markdown_table:
        structure_score = round(0.5 + 0.5 * cell_recall, 3)
    else:
        # Flat text dump loses structure completely
        structure_score = round(0.25 * cell_recall, 3)

    return {
        "is_markdown_table": is_markdown_table,
        "cells_total": total_cells,
        "cells_detected": cells_found,
        "cell_recall": cell_recall,
        "table_structure_score": structure_score,
    }

def evaluate_formula_fidelity(gt_formulas: List[str], expected_terms: List[str], pred_text: str) -> Dict[str, Any]:
    """
    Evaluates mathematical formula extraction fidelity (LaTeX structure and mathematical symbols).
    """
    has_latex_formatting = any(marker in pred_text for marker in ["\\frac", "\\sum", "\\lambda", "\\theta", "\\exp", "$"])
    terms_found = sum(1 for term in expected_terms if term.lower() in pred_text.lower())
    term_accuracy = round(terms_found / len(expected_terms), 3) if expected_terms else 0.0

    return {
        "has_latex": has_latex_formatting,
        "terms_detected": terms_found,
        "terms_total": len(expected_terms),
        "term_accuracy": term_accuracy,
        "formula_fidelity_score": round((0.6 if has_latex_formatting else 0.1) + 0.4 * term_accuracy, 3)
    }
