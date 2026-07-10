"""Portable JSON and Markdown artefacts for a search run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _text(value: object) -> str:
    return str(value).replace("\n", " ").strip()


def save_report_tool(
    query: str,
    interpreted_intent: str,
    filters: dict[str, Any],
    results: list[dict[str, Any]],
    output_path: str | Path = "outputs/search_report.md",
) -> str:
    """Save a human-readable report and return its project-relative path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image Extractor Agent — Search Report",
        "",
        "## User Query",
        "",
        query,
        "",
        "## Interpreted Intent",
        "",
        interpreted_intent,
        "",
        "## Filters",
        "",
        f"- Location: {filters.get('location') or 'None'}",
        f"- Date: {filters.get('date') or 'None'}",
        "",
        "## Top Results",
        "",
    ]
    if not results:
        lines.append("No matching images were found.")
    for result in results:
        lines.extend(
            [
                f"### {result['rank']}. {_text(result['file_name'])}",
                "",
                f"- Path: `{_text(result['image_path'])}`",
                f"- Location: {_text(result.get('location', 'unknown'))}",
                f"- Date: {_text(result.get('date', 'unknown'))}",
                f"- Size: {_text(result.get('width', '?'))}x{_text(result.get('height', '?'))}",
                f"- Similarity Score: {float(result['similarity_score']):.4f}",
                f"- Generated Caption: {_text(result.get('generated_caption', 'No caption generated.'))}",
                f"- Tags: {', '.join(map(_text, result.get('tags', []))) or 'None'}",
                f"- Explanation: {_text(result.get('explanation', ''))}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.as_posix()


def save_results_json(payload: dict[str, Any], output_path: str | Path = "outputs/search_results.json") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path.as_posix()
