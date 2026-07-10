"""Write JSON + Markdown search reports."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def save_report(answer: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "search_results.json"
    md_path = OUT_DIR / "search_report.md"

    json_path.write_text(json.dumps(answer, indent=2, ensure_ascii=False))

    lines = ["# Image Extractor Agent — Search Report", ""]
    lines += ["## User Query", "", answer["query"], ""]
    lines += ["## Interpreted Intent", "", answer["interpreted_intent"], ""]
    lines += ["## Filters", "",
              f"- Date: {answer['filters'].get('date') or 'None'}", ""]
    lines += ["## Top Results", ""]
    if not answer["results"]:
        lines.append("_No images matched._")
    for r in answer["results"]:
        lines += [
            f"### {r['rank']}. {r['file_name']}",
            "",
            f"- Path: `{r['image_path']}`",
            f"- Date: {r['date']}",
            f"- Size: {r['width']}x{r['height']}",
            f"- Similarity Score: {r['similarity_score']}",
            f"- Generated Caption: {r.get('generated_caption', 'N/A')}",
            f"- Tags: {', '.join(r.get('tags', [])) or 'N/A'}",
            f"- Explanation: {r.get('explanation', '')}",
            "",
        ]

    md_path.write_text("\n".join(lines))
    return {"json": str(json_path), "md": str(md_path)}
