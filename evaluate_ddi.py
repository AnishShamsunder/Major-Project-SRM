"""
evaluate_ddi.py
---------------
Evaluation harness for the RAG + LLM response path.

What this script does:
1. Uses a hard-coded ground-truth list of drug pairs.
2. Calls the live /ask endpoint for each pair.
3. Parses the LLM response into a binary prediction (interaction yes/no).
4. Compares predictions vs labels to build a confusion matrix.
5. Reports precision, recall, F1-score, and accuracy.

This script is stdlib-only and evaluates actual generated answers.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass(frozen=True)
class GroundTruthCase:
    drug1: str
    drug2: str
    interacts: bool
    reason: str = ""  # Optional rationale for human-readable reporting.


# Small manually curated benchmark set.
# Positives are examples already highlighted in project docs.
# Negatives are intended non-DDI controls for sanity-checking.
GROUND_TRUTHS: list[GroundTruthCase] = [
    GroundTruthCase("Warfarin", "Aspirin", True, "Known increased bleeding risk"),
    GroundTruthCase("Simvastatin", "Amlodipine", True, "Known statin level increase risk"),
    GroundTruthCase("Sertraline", "Ibuprofen", True, "Known GI bleeding risk"),
    GroundTruthCase("Clopidogrel", "Omeprazole", True, "Known reduced antiplatelet efficacy"),
    GroundTruthCase("Metformin", "Ciprofloxacin", True, "Known blood glucose effect risk"),
    GroundTruthCase("Warfarin", "Rifampicin", True, "Known warfarin concentration reduction"),
    GroundTruthCase("Acetaminophen", "Loratadine", False, "Control pair"),
    GroundTruthCase("Amoxicillin", "Cetirizine", False, "Control pair"),
    GroundTruthCase("Vitamin C", "Loratadine", False, "Control pair"),
    GroundTruthCase("Saline", "Paracetamol", False, "Control pair"),
    GroundTruthCase("Calcium Carbonate", "Topical Aloe Vera", False, "Control pair"),
    GroundTruthCase("Normal Saline", "Cetirizine", False, "Control pair"),
]


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def ask_rag(api_base: str, drug1: str, drug2: str, top_k: int) -> dict:
    query = f"What happens if I take {drug1} and {drug2} together?"
    payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    req = urllib_request.Request(
        f"{api_base.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise RuntimeError("Unexpected non-object JSON from /ask")
            return data
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"/ask returned HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach backend at {api_base}. Start API first: uvicorn main:app --port 8000"
        ) from exc


def _extract_together_section(answer: str) -> str:
    pattern = re.compile(
        r"\*\*What happens when both are taken together:\*\*\s*([\s\S]*?)(?=\*\*What happens in the body:|$)",
        re.IGNORECASE,
    )
    match = pattern.search(answer or "")
    return (match.group(1) if match else "").strip()


def detect_interaction_from_llm(answer: str) -> tuple[bool, float, str]:
    """
    Convert free-form LLM answer into binary prediction.

    Hard-coded safety-aware rule:
    1) If answer contains explicit insufficiency / out-of-scope text -> negative.
    2) Else inspect section 1 ("taken together"):
       - non-empty bullet-like content with interaction cues -> positive.
       - otherwise negative.
    """
    text = _normalize(answer)

    hard_negative_markers = [
        "the knowledge base does not contain enough information about this combination",
        "this question is outside the scope of the drug interaction knowledge base",
    ]
    if any(marker in text for marker in hard_negative_markers):
        return False, 0.95, "hard_negative_marker"

    together = _extract_together_section(answer)
    together_n = _normalize(together)
    if not together_n:
        return False, 0.6, "missing_together_section"

    interaction_cues = [
        "interact",
        "interaction",
        "increases",
        "decreases",
        "reduced",
        "elevated",
        "risk",
        "bleeding",
        "toxicity",
        "serum concentration",
        "efficacy",
        "adverse",
    ]
    hits = sum(1 for cue in interaction_cues if cue in together_n)
    if hits > 0:
        confidence = min(0.99, 0.65 + 0.05 * hits)
        return True, confidence, f"together_section_cues={hits}"

    return False, 0.55, "no_interaction_cues_in_together_section"


def evaluate(
    cases: list[GroundTruthCase],
    *,
    api_base: str,
    top_k: int,
) -> dict:
    tp = fp = tn = fn = 0
    details: list[dict] = []

    for case in cases:
        rag = ask_rag(api_base=api_base, drug1=case.drug1, drug2=case.drug2, top_k=top_k)
        answer = str(rag.get("answer", ""))
        pred, support_score, support_rule = detect_interaction_from_llm(answer)
        truth = case.interacts

        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and not truth:
            tn += 1
        else:
            fn += 1

        details.append(
            {
                "drug1": case.drug1,
                "drug2": case.drug2,
                "truth": truth,
                "pred": pred,
                "support_score": round(support_score, 6),
                "support_rule": support_rule,
                "reason": case.reason,
                "answer_excerpt": (answer[:240] + "...") if len(answer) > 240 else answer,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(cases) if cases else 0.0

    return {
        "n_cases": len(cases),
        "detector": "rag_llm_response_parser",
        "api_base": api_base,
        "top_k": top_k,
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        },
        "metrics": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "accuracy": round(accuracy, 6),
        },
        "details": details,
    }


def print_report(report: dict) -> None:
    cm = report["confusion_matrix"]
    m = report["metrics"]

    print("\nDDI Detection Evaluation")
    print("=" * 30)
    print(f"Cases: {report['n_cases']}")
    print(f"Detector: {report['detector']}")
    print(f"API base: {report['api_base']}")
    print(f"top_k: {report['top_k']}")

    print("\nConfusion Matrix")
    print("----------------")
    print("                 Predicted +    Predicted -")
    print(f"Actual +         {cm['TP']:<13}{cm['FN']}")
    print(f"Actual -         {cm['FP']:<13}{cm['TN']}")

    print("\nMetrics")
    print("-------")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1-score:  {m['f1']:.4f}")
    print(f"Accuracy:  {m['accuracy']:.4f}")

    print("\nPer-case results")
    print("----------------")
    for row in report["details"]:
        pair = f"{row['drug1']} + {row['drug2']}"
        print(
            f"- {pair}: truth={row['truth']} pred={row['pred']} "
            f"score={row['support_score']:.4f} rule={row['support_rule']}"
        )


def _write_metrics_csv(report: dict, out_dir: Path) -> Path:
    path = out_dir / "metrics_summary.csv"
    metrics = report["metrics"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["precision", metrics["precision"]])
        writer.writerow(["recall", metrics["recall"]])
        writer.writerow(["f1", metrics["f1"]])
        writer.writerow(["accuracy", metrics["accuracy"]])
    return path


def _write_confusion_csv(report: dict, out_dir: Path) -> Path:
    path = out_dir / "confusion_matrix.csv"
    cm = report["confusion_matrix"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual", "predicted_positive", "predicted_negative"])
        writer.writerow(["positive", cm["TP"], cm["FN"]])
        writer.writerow(["negative", cm["FP"], cm["TN"]])
    return path


def _write_per_case_csv(report: dict, out_dir: Path) -> Path:
    path = out_dir / "per_case_results.csv"
    rows = report["details"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "drug1",
                "drug2",
                "truth",
                "prediction",
                "support_score",
                "support_rule",
                "reason",
                "answer_excerpt",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["drug1"],
                    row["drug2"],
                    row["truth"],
                    row["pred"],
                    row["support_score"],
                    row["support_rule"],
                    row["reason"],
                    row["answer_excerpt"],
                ]
            )
    return path


def _write_tables_markdown(report: dict, out_dir: Path) -> Path:
    path = out_dir / "paper_tables.md"
    cm = report["confusion_matrix"]
    m = report["metrics"]

    lines: list[str] = []
    lines.append("# DDI Evaluation Tables")
    lines.append("")
    lines.append("## Table 1. Confusion Matrix")
    lines.append("")
    lines.append("| Actual \\ Predicted | Positive | Negative |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Positive | {cm['TP']} | {cm['FN']} |")
    lines.append(f"| Negative | {cm['FP']} | {cm['TN']} |")
    lines.append("")
    lines.append("## Table 2. Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Precision | {m['precision']:.4f} |")
    lines.append(f"| Recall | {m['recall']:.4f} |")
    lines.append(f"| F1-score | {m['f1']:.4f} |")
    lines.append(f"| Accuracy | {m['accuracy']:.4f} |")
    lines.append("")
    lines.append("## Table 3. Per-case Results")
    lines.append("")
    lines.append("| Drug 1 | Drug 2 | Ground Truth | Prediction | Support Rule |")
    lines.append("|---|---|---:|---:|---|")
    for row in report["details"]:
        lines.append(
            f"| {row['drug1']} | {row['drug2']} | {int(bool(row['truth']))} | {int(bool(row['pred']))} | {row['support_rule']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_confusion_svg(report: dict, out_dir: Path) -> Path:
    path = out_dir / "confusion_matrix.svg"
    cm = report["confusion_matrix"]
    cells = [cm["TP"], cm["FN"], cm["FP"], cm["TN"]]
    max_cell = max(cells) if cells else 1

    def color(value: int) -> str:
        # Blue intensity scale for paper-friendly readability.
        intensity = int(245 - (160 * (value / max_cell)))
        intensity = max(70, min(245, intensity))
        return f"rgb({intensity},{intensity},{255})"

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"620\" viewBox=\"0 0 900 620\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"450\" y=\"55\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"30\" font-weight=\"bold\">Confusion Matrix</text>
  <text x=\"450\" y=\"92\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">RAG-LLM DDI Detection</text>

  <text x=\"450\" y=\"145\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"22\">Predicted Label</text>
  <text x=\"115\" y=\"355\" text-anchor=\"middle\" transform=\"rotate(-90 115,355)\" font-family=\"Times New Roman, serif\" font-size=\"22\">Actual Label</text>

  <text x=\"390\" y=\"190\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">Positive</text>
  <text x=\"590\" y=\"190\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">Negative</text>

  <text x=\"250\" y=\"300\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">Positive</text>
  <text x=\"250\" y=\"500\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">Negative</text>

  <rect x=\"300\" y=\"220\" width=\"180\" height=\"180\" fill=\"{color(cm['TP'])}\" stroke=\"#222\" stroke-width=\"2\"/>
  <rect x=\"500\" y=\"220\" width=\"180\" height=\"180\" fill=\"{color(cm['FN'])}\" stroke=\"#222\" stroke-width=\"2\"/>
  <rect x=\"300\" y=\"420\" width=\"180\" height=\"180\" fill=\"{color(cm['FP'])}\" stroke=\"#222\" stroke-width=\"2\"/>
  <rect x=\"500\" y=\"420\" width=\"180\" height=\"180\" fill=\"{color(cm['TN'])}\" stroke=\"#222\" stroke-width=\"2\"/>

  <text x=\"390\" y=\"310\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"24\" font-weight=\"bold\">TP</text>
  <text x=\"390\" y=\"345\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"32\">{cm['TP']}</text>

  <text x=\"590\" y=\"310\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"24\" font-weight=\"bold\">FN</text>
  <text x=\"590\" y=\"345\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"32\">{cm['FN']}</text>

  <text x=\"390\" y=\"510\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"24\" font-weight=\"bold\">FP</text>
  <text x=\"390\" y=\"545\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"32\">{cm['FP']}</text>

  <text x=\"590\" y=\"510\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"24\" font-weight=\"bold\">TN</text>
  <text x=\"590\" y=\"545\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"32\">{cm['TN']}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def _write_metrics_svg(report: dict, out_dir: Path) -> Path:
    path = out_dir / "metrics_bar_chart.svg"
    m = report["metrics"]
    series = [
        ("Precision", float(m["precision"])),
        ("Recall", float(m["recall"])),
        ("F1", float(m["f1"])),
        ("Accuracy", float(m["accuracy"])),
    ]

    chart_x = 110
    chart_y = 120
    chart_w = 700
    chart_h = 360
    bar_w = 120
    gap = 50
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    bars = []
    for i, (label, value) in enumerate(series):
        x = chart_x + i * (bar_w + gap)
        h = int(chart_h * max(0.0, min(1.0, value)))
        y = chart_y + (chart_h - h)
        bars.append(
            f"<rect x=\"{x}\" y=\"{y}\" width=\"{bar_w}\" height=\"{h}\" fill=\"{colors[i]}\" opacity=\"0.88\"/>"
        )
        bars.append(
            f"<text x=\"{x + bar_w / 2}\" y=\"{y - 10}\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">{value:.3f}</text>"
        )
        bars.append(
            f"<text x=\"{x + bar_w / 2}\" y=\"{chart_y + chart_h + 34}\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">{label}</text>"
        )

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"920\" height=\"600\" viewBox=\"0 0 920 600\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"460\" y=\"52\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"30\" font-weight=\"bold\">DDI Evaluation Metrics</text>
  <text x=\"460\" y=\"84\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">Precision, Recall, F1-score, Accuracy</text>

  <line x1=\"{chart_x}\" y1=\"{chart_y}\" x2=\"{chart_x}\" y2=\"{chart_y + chart_h}\" stroke=\"#222\" stroke-width=\"2\"/>
  <line x1=\"{chart_x}\" y1=\"{chart_y + chart_h}\" x2=\"{chart_x + chart_w}\" y2=\"{chart_y + chart_h}\" stroke=\"#222\" stroke-width=\"2\"/>

  <line x1=\"{chart_x}\" y1=\"{chart_y}\" x2=\"{chart_x + chart_w}\" y2=\"{chart_y}\" stroke=\"#ddd\" stroke-width=\"1\"/>
  <line x1=\"{chart_x}\" y1=\"{chart_y + chart_h * 0.25}\" x2=\"{chart_x + chart_w}\" y2=\"{chart_y + chart_h * 0.25}\" stroke=\"#eee\" stroke-width=\"1\"/>
  <line x1=\"{chart_x}\" y1=\"{chart_y + chart_h * 0.5}\" x2=\"{chart_x + chart_w}\" y2=\"{chart_y + chart_h * 0.5}\" stroke=\"#eee\" stroke-width=\"1\"/>
  <line x1=\"{chart_x}\" y1=\"{chart_y + chart_h * 0.75}\" x2=\"{chart_x + chart_w}\" y2=\"{chart_y + chart_h * 0.75}\" stroke=\"#eee\" stroke-width=\"1\"/>

  <text x=\"85\" y=\"{chart_y + 8}\" font-family=\"Times New Roman, serif\" font-size=\"18\">1.0</text>
  <text x=\"85\" y=\"{chart_y + chart_h * 0.25 + 6}\" font-family=\"Times New Roman, serif\" font-size=\"18\">0.75</text>
  <text x=\"85\" y=\"{chart_y + chart_h * 0.5 + 6}\" font-family=\"Times New Roman, serif\" font-size=\"18\">0.50</text>
  <text x=\"85\" y=\"{chart_y + chart_h * 0.75 + 6}\" font-family=\"Times New Roman, serif\" font-size=\"18\">0.25</text>
  <text x=\"85\" y=\"{chart_y + chart_h + 6}\" font-family=\"Times New Roman, serif\" font-size=\"18\">0.0</text>

  {''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def export_paper_artifacts(report: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics_csv": _write_metrics_csv(report, out_dir),
        "confusion_csv": _write_confusion_csv(report, out_dir),
        "per_case_csv": _write_per_case_csv(report, out_dir),
        "tables_markdown": _write_tables_markdown(report, out_dir),
        "confusion_svg": _write_confusion_svg(report, out_dir),
        "metrics_svg": _write_metrics_svg(report, out_dir),
    }

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs["report_json"] = json_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DDI detection from actual RAG LLM responses")
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000",
        help="Base URL for backend API",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Context chunks for /ask")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="evaluation_outputs",
        help="Directory to save paper-ready tables and images",
    )
    args = parser.parse_args()

    report = evaluate(GROUND_TRUTHS, api_base=args.api_base, top_k=args.top_k)
    print_report(report)

    outputs = export_paper_artifacts(report, Path(args.out_dir))
    print("\nSaved artifacts")
    print("---------------")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
