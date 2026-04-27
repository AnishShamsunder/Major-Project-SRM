"""
evaluate_ddi_details.py
-----------------------
Detail-level evaluation for DDI RAG responses.

Goal:
- Use ground truth directly from the dataset (Drug 1, Drug 2, Interaction Description).
- Ask the live RAG /ask endpoint for each pair.
- Compare the LLM's "taken together" explanation with the dataset description.
- Mark each case as MATCH / NO_MATCH and export paper-ready artifacts.

This script is stdlib-only.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import statistics
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass(frozen=True)
class DetailCase:
    drug1: str
    drug2: str
    ground_truth: str


STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "as",
    "from", "at", "into", "than", "then", "both", "taken", "together", "may", "can",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize(text))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _extract_together_section(answer: str) -> str:
    pattern = re.compile(
        r"\*\*What happens when both are taken together:\*\*\s*([\s\S]*?)(?=\*\*What happens in the body:|$)",
        re.IGNORECASE,
    )
    match = pattern.search(answer or "")
    section = (match.group(1) if match else "").strip()
    return section if section else (answer or "")


def _extract_cues(text: str) -> set[str]:
    t = _normalize(text)
    cues: set[str] = set()

    if any(w in t for w in ["increase", "increases", "increased", "elevated", "elevation"]):
        cues.add("increase")
    if any(w in t for w in ["decrease", "decreases", "decreased", "reduced", "reduction", "lower"]):
        cues.add("decrease")
    if "bleeding" in t or "hemorrhage" in t:
        cues.add("bleeding")
    if "toxicity" in t or "toxic" in t:
        cues.add("toxicity")
    if "efficacy" in t or "effective" in t:
        cues.add("efficacy")
    if "serum concentration" in t or "plasma concentration" in t or "drug level" in t:
        cues.add("concentration")

    return cues


def compare_details(ground_truth: str, llm_text: str, threshold: float) -> dict:
    gt_tokens = set(_tokenize(ground_truth))
    pred_tokens = set(_tokenize(llm_text))
    overlap = gt_tokens & pred_tokens

    token_precision = len(overlap) / len(pred_tokens) if pred_tokens else 0.0
    token_recall = len(overlap) / len(gt_tokens) if gt_tokens else 0.0
    token_f1 = (
        2 * token_precision * token_recall / (token_precision + token_recall)
        if (token_precision + token_recall)
        else 0.0
    )

    gt_cues = _extract_cues(ground_truth)
    pred_cues = _extract_cues(llm_text)
    cue_recall = len(gt_cues & pred_cues) / len(gt_cues) if gt_cues else 1.0

    combined_score = 0.7 * token_f1 + 0.3 * cue_recall
    is_match = combined_score >= threshold

    return {
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
        "cue_recall": cue_recall,
        "combined_score": combined_score,
        "is_match": is_match,
        "gt_cues": sorted(gt_cues),
        "pred_cues": sorted(pred_cues),
    }


def load_cases_from_dataset(csv_path: Path, max_cases: int, sampling: str, seed: int) -> list[DetailCase]:
    rows: list[DetailCase] = []

    with csv_path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            d1 = (row.get("Drug 1") or "").strip()
            d2 = (row.get("Drug 2") or "").strip()
            desc = (row.get("Interaction Description") or "").strip()
            if d1 and d2 and desc:
                rows.append(DetailCase(drug1=d1, drug2=d2, ground_truth=desc))

    if not rows:
        raise RuntimeError("No valid rows found in dataset CSV.")

    if sampling == "random":
        rng = random.Random(seed)
        rng.shuffle(rows)

    return rows[:max_cases]


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
        with urllib_request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"/ask HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach backend at {api_base}. Start API first: uvicorn main:app --port 8000"
        ) from exc


def evaluate_detail_accuracy(
    cases: list[DetailCase],
    *,
    api_base: str,
    top_k: int,
    match_threshold: float,
) -> dict:
    details: list[dict] = []
    match_count = 0

    for idx, case in enumerate(cases, start=1):
        rag = ask_rag(api_base=api_base, drug1=case.drug1, drug2=case.drug2, top_k=top_k)
        answer = str(rag.get("answer", ""))
        together = _extract_together_section(answer)

        comp = compare_details(case.ground_truth, together, match_threshold)
        if comp["is_match"]:
            match_count += 1

        details.append(
            {
                "case_id": idx,
                "drug1": case.drug1,
                "drug2": case.drug2,
                "ground_truth": case.ground_truth,
                "llm_together_section": together,
                "token_precision": round(comp["token_precision"], 6),
                "token_recall": round(comp["token_recall"], 6),
                "token_f1": round(comp["token_f1"], 6),
                "cue_recall": round(comp["cue_recall"], 6),
                "combined_score": round(comp["combined_score"], 6),
                "match": bool(comp["is_match"]),
                "gt_cues": ";".join(comp["gt_cues"]),
                "pred_cues": ";".join(comp["pred_cues"]),
            }
        )

    n = len(details)
    token_precision_avg = statistics.mean(d["token_precision"] for d in details) if details else 0.0
    token_recall_avg = statistics.mean(d["token_recall"] for d in details) if details else 0.0
    token_f1_avg = statistics.mean(d["token_f1"] for d in details) if details else 0.0
    cue_recall_avg = statistics.mean(d["cue_recall"] for d in details) if details else 0.0
    combined_avg = statistics.mean(d["combined_score"] for d in details) if details else 0.0
    match_rate = (match_count / n) if n else 0.0

    return {
        "n_cases": n,
        "api_base": api_base,
        "top_k": top_k,
        "match_threshold": match_threshold,
        "summary": {
            "detail_match_rate": round(match_rate, 6),
            "avg_token_precision": round(token_precision_avg, 6),
            "avg_token_recall": round(token_recall_avg, 6),
            "avg_token_f1": round(token_f1_avg, 6),
            "avg_cue_recall": round(cue_recall_avg, 6),
            "avg_combined_score": round(combined_avg, 6),
            "matches": match_count,
            "non_matches": n - match_count,
        },
        "details": details,
    }


def _write_summary_csv(report: dict, out_dir: Path) -> Path:
    path = out_dir / "detail_accuracy_summary.csv"
    s = report["summary"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in [
            "detail_match_rate",
            "avg_token_precision",
            "avg_token_recall",
            "avg_token_f1",
            "avg_cue_recall",
            "avg_combined_score",
            "matches",
            "non_matches",
        ]:
            writer.writerow([key, s[key]])
    return path


def _write_per_case_csv(report: dict, out_dir: Path) -> Path:
    path = out_dir / "detail_accuracy_per_case.csv"
    rows = report["details"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "case_id", "drug1", "drug2", "match", "token_precision", "token_recall",
            "token_f1", "cue_recall", "combined_score", "gt_cues", "pred_cues",
            "ground_truth", "llm_together_section",
        ])
        for r in rows:
            writer.writerow([
                r["case_id"], r["drug1"], r["drug2"], r["match"], r["token_precision"],
                r["token_recall"], r["token_f1"], r["cue_recall"], r["combined_score"],
                r["gt_cues"], r["pred_cues"], r["ground_truth"], r["llm_together_section"],
            ])
    return path


def _write_markdown_tables(report: dict, out_dir: Path) -> Path:
    path = out_dir / "detail_accuracy_tables.md"
    s = report["summary"]
    lines: list[str] = []
    lines.append("# DDI Detail Accuracy Tables")
    lines.append("")
    lines.append("## Table 1. Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Detail Match Rate | {s['detail_match_rate']:.4f} |")
    lines.append(f"| Avg Token Precision | {s['avg_token_precision']:.4f} |")
    lines.append(f"| Avg Token Recall | {s['avg_token_recall']:.4f} |")
    lines.append(f"| Avg Token F1 | {s['avg_token_f1']:.4f} |")
    lines.append(f"| Avg Cue Recall | {s['avg_cue_recall']:.4f} |")
    lines.append(f"| Avg Combined Score | {s['avg_combined_score']:.4f} |")
    lines.append(f"| Matches | {s['matches']} |")
    lines.append(f"| Non-matches | {s['non_matches']} |")
    lines.append("")
    lines.append("## Table 2. Per-case Match Judgement")
    lines.append("")
    lines.append("| Case | Drug 1 | Drug 2 | Match | Token F1 | Cue Recall | Combined |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for r in report["details"]:
        lines.append(
            f"| {r['case_id']} | {r['drug1']} | {r['drug2']} | {int(r['match'])} | {r['token_f1']:.4f} | {r['cue_recall']:.4f} | {r['combined_score']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_match_svg(report: dict, out_dir: Path) -> Path:
    path = out_dir / "detail_match_distribution.svg"
    s = report["summary"]
    matches = int(s["matches"])
    non_matches = int(s["non_matches"])
    total = max(1, matches + non_matches)
    match_w = int(640 * (matches / total))
    non_match_w = 640 - match_w

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"320\" viewBox=\"0 0 900 320\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>
  <text x=\"450\" y=\"48\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"30\" font-weight=\"bold\">Detail Match Distribution</text>
  <text x=\"450\" y=\"82\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"20\">Ground Truth Interaction Details vs LLM Response</text>

  <rect x=\"130\" y=\"140\" width=\"{match_w}\" height=\"64\" fill=\"#2ca02c\"/>
  <rect x=\"{130 + match_w}\" y=\"140\" width=\"{non_match_w}\" height=\"64\" fill=\"#d62728\"/>

  <text x=\"150\" y=\"130\" font-family=\"Times New Roman, serif\" font-size=\"20\">Matches: {matches}</text>
  <text x=\"620\" y=\"130\" font-family=\"Times New Roman, serif\" font-size=\"20\">Non-matches: {non_matches}</text>
  <text x=\"450\" y=\"250\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"22\">Match Rate = {s['detail_match_rate']:.3f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def _write_per_case_f1_svg(report: dict, out_dir: Path) -> Path:
    path = out_dir / "per_case_token_f1.svg"
    rows = report["details"]
    n = len(rows)
    width = 1100
    height = 380
    left = 70
    bottom = 300
    chart_h = 220
    bar_w = max(8, int((width - 2 * left) / max(1, n) - 6))

    bars = []
    labels = []
    for i, r in enumerate(rows):
        val = float(r["token_f1"])
        x = left + i * (bar_w + 6)
        h = int(chart_h * max(0.0, min(1.0, val)))
        y = bottom - h
        color = "#1f77b4" if r["match"] else "#ff7f0e"
        bars.append(f"<rect x=\"{x}\" y=\"{y}\" width=\"{bar_w}\" height=\"{h}\" fill=\"{color}\"/>")
        labels.append(f"<text x=\"{x + bar_w/2}\" y=\"{bottom + 20}\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"10\">{r['case_id']}</text>")

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>
  <text x=\"{width/2}\" y=\"36\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"24\" font-weight=\"bold\">Per-case Token F1 vs Ground Truth Detail</text>
  <line x1=\"{left}\" y1=\"{bottom-chart_h}\" x2=\"{left}\" y2=\"{bottom}\" stroke=\"#222\" stroke-width=\"2\"/>
  <line x1=\"{left}\" y1=\"{bottom}\" x2=\"{width-left}\" y2=\"{bottom}\" stroke=\"#222\" stroke-width=\"2\"/>
  <text x=\"20\" y=\"{bottom-chart_h+6}\" font-family=\"Times New Roman, serif\" font-size=\"14\">1.0</text>
  <text x=\"20\" y=\"{bottom-chart_h/2+6}\" font-family=\"Times New Roman, serif\" font-size=\"14\">0.5</text>
  <text x=\"20\" y=\"{bottom+6}\" font-family=\"Times New Roman, serif\" font-size=\"14\">0.0</text>
  {''.join(bars)}
  {''.join(labels)}
  <text x=\"{width/2}\" y=\"{height-12}\" text-anchor=\"middle\" font-family=\"Times New Roman, serif\" font-size=\"14\">Case ID (see per-case table)</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def export_artifacts(report: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_csv": _write_summary_csv(report, out_dir),
        "per_case_csv": _write_per_case_csv(report, out_dir),
        "tables_markdown": _write_markdown_tables(report, out_dir),
        "match_distribution_svg": _write_match_svg(report, out_dir),
        "per_case_f1_svg": _write_per_case_f1_svg(report, out_dir),
    }
    report_path = out_dir / "detail_accuracy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs["report_json"] = report_path
    return outputs


def print_console_summary(report: dict) -> None:
    s = report["summary"]
    print("\nDDI Detail Accuracy Evaluation")
    print("=" * 34)
    print(f"Cases: {report['n_cases']}")
    print(f"Detail match threshold: {report['match_threshold']}")
    print(f"Match rate: {s['detail_match_rate']:.4f} ({s['matches']}/{report['n_cases']})")
    print(f"Avg token precision: {s['avg_token_precision']:.4f}")
    print(f"Avg token recall:    {s['avg_token_recall']:.4f}")
    print(f"Avg token F1:        {s['avg_token_f1']:.4f}")
    print(f"Avg cue recall:      {s['avg_cue_recall']:.4f}")
    print(f"Avg combined score:  {s['avg_combined_score']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether LLM interaction details match dataset ground truth")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--top-k", type=int, default=5, help="Context chunks for /ask")
    parser.add_argument("--dataset", type=str, default="data/raw/drug_drug_interactions.csv", help="Path to DrugBank DDI CSV")
    parser.add_argument("--max-cases", type=int, default=20, help="Number of dataset pairs to evaluate")
    parser.add_argument("--sampling", type=str, choices=["head", "random"], default="head", help="Case selection strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when sampling=random")
    parser.add_argument("--threshold", type=float, default=0.45, help="Combined score threshold for MATCH")
    parser.add_argument("--out-dir", type=str, default="evaluation_outputs/detail_accuracy", help="Output directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cases = load_cases_from_dataset(
        csv_path=dataset_path,
        max_cases=args.max_cases,
        sampling=args.sampling,
        seed=args.seed,
    )

    report = evaluate_detail_accuracy(
        cases,
        api_base=args.api_base,
        top_k=args.top_k,
        match_threshold=args.threshold,
    )
    print_console_summary(report)

    outputs = export_artifacts(report, Path(args.out_dir))
    print("\nSaved artifacts")
    print("---------------")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
