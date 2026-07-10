"""Query parsing, retrieval, metadata filtering, and deterministic explanations."""

from __future__ import annotations

import calendar
import re
from typing import Any

from .embedding_index import LocalEmbeddingIndex


MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}


def parse_query_tool(query: str, known_locations: list[str] | None = None) -> dict[str, Any]:
    """Extract basic explicit location/date filters while preserving visual intent."""
    if not query or not query.strip():
        raise ValueError("Please enter a non-empty image search query.")
    visual_query = query.strip()
    location = None
    for candidate in known_locations or []:
        if re.search(rf"\b{re.escape(candidate)}\b", query, flags=re.IGNORECASE):
            location = candidate
            visual_query = re.sub(rf"\b(?:in|from|taken in)\s+{re.escape(candidate)}\b", "", visual_query, flags=re.IGNORECASE)
            break

    date_filter = None
    exact_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
    if exact_date:
        date_filter = exact_date.group(1)
        visual_query = visual_query.replace(date_filter, "")
    else:
        month_match = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(20\d{2})\b", query, flags=re.IGNORECASE)
        if month_match:
            date_filter = f"{month_match.group(2)}-{MONTHS[month_match.group(1).lower()]:02d}"
            visual_query = re.sub(month_match.group(0), "", visual_query, flags=re.IGNORECASE)
        else:
            year_match = re.search(r"\b(20\d{2})\b", query)
            if year_match:
                date_filter = year_match.group(1)
                visual_query = visual_query.replace(date_filter, "")

    visual_query = re.sub(r"\s+", " ", visual_query).strip(" ,.-")
    return {
        "visual_query": visual_query or query.strip(),
        "location": location,
        "date": date_filter,
        "top_k": 5,
    }


def search_images_tool(
    index: LocalEmbeddingIndex,
    records: list[dict[str, Any]],
    visual_query: str,
    top_k: int | None = 5,
) -> list[dict[str, Any]]:
    """Score records by cosine similarity and return the best candidates."""
    by_id = {str(record["image_id"]): record for record in records}
    scores = index.similarity_scores(visual_query)
    ranked = sorted(zip(index.image_ids, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
    if top_k is not None:
        ranked = ranked[: max(top_k, 0)]
    return [
        {**dict(by_id[image_id]), "similarity_score": round(float(score), 4)}
        for image_id, score in ranked
        if image_id in by_id
    ]


def metadata_filter_tool(results: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep results matching explicitly parsed location/date metadata filters."""
    location = filters.get("location")
    date_filter = filters.get("date")
    filtered = results
    if location:
        filtered = [item for item in filtered if str(item.get("location", "")).casefold() == str(location).casefold()]
    if date_filter:
        filtered = [item for item in filtered if str(item.get("date", "")).startswith(str(date_filter))]
    return filtered


def _query_terms(query: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[a-zA-Z]{3,}", query)}


def explain_results_tool(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add concise, auditable explanations without inventing unseen content."""
    terms = _query_terms(query)
    explained: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        item = dict(result)
        tags = [*item.get("objects", []), *item.get("activities", []), *item.get("visual_attributes", [])]
        searchable = " ".join([str(item.get("generated_caption", "")), str(item.get("scene", "")), *map(str, tags)]).lower()
        matched_terms = sorted(term for term in terms if term in searchable)
        pieces = []
        if matched_terms:
            pieces.append("visual description mentions " + ", ".join(matched_terms))
        else:
            pieces.append("its generated visual description is the closest semantic match")
        if item.get("location") and str(item["location"]).casefold() in query.casefold():
            pieces.append(f"location matches {item['location']}")
        if item.get("date") and str(item["date"])[:7] in query:
            pieces.append(f"date matches {item['date']}")
        item["rank"] = rank
        item["tags"] = tags
        item["explanation"] = "This image matches because " + " and ".join(pieces) + "."
        explained.append(item)
    return explained
