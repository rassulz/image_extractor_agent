"""Agent tools: parse query, metadata filter, CLIP search, explain."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import embedding_index as idx
from . import vision_extractor

ROOT = Path(__file__).resolve().parents[1]

MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def parse_query_tool(query: str, top_k: int = 5) -> dict:
    """Rule-based parse into visual query + metadata filters (date only).

    Location is intentionally not parsed: Flickr8k has no real location metadata.
    """
    location = None
    date_prefix = None
    # e.g. "July 2026" -> 2026-07 ; "2026-07-05" -> exact
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", query)
    if m:
        date_prefix = m.group(0)
    else:
        m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(20\d{2})\b", query, re.IGNORECASE)
        if m:
            date_prefix = f"{m.group(2)}-{MONTHS[m.group(1).lower()]}"
        else:
            m = re.search(r"\b(20\d{2})\b", query)
            if m:
                date_prefix = m.group(1)

    # Strip filter phrases from the visual query so CLIP sees clean visual text.
    visual = query
    visual = re.sub(r"\bfind (images?|photos?)\b", "", visual, flags=re.IGNORECASE)
    visual = re.sub(r"\bshow me\b", "", visual, flags=re.IGNORECASE)
    visual = re.sub(r"\bwith\b", "", visual, flags=re.IGNORECASE)
    visual = re.sub(r"\s+", " ", visual).strip(" .,")

    intent = f"Search for images of: {visual or query}"
    if location:
        intent += f" | location={location}"
    if date_prefix:
        intent += f" | date~{date_prefix}"

    return {
        "visual_query": visual or query,
        "location": location,
        "date": date_prefix,
        "top_k": top_k,
        "interpreted_intent": intent,
    }


def metadata_filter_tool(df: pd.DataFrame, filters: dict) -> list[str]:
    """Return candidate image_ids matching the metadata filters (date only)."""
    mask = pd.Series(True, index=df.index)
    if filters.get("date"):
        mask &= df["date"].astype(str).str.startswith(filters["date"])
    return df.loc[mask, "image_id"].tolist()


def search_images_tool(
    visual_query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    image_ids: list[str],
    candidate_ids: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Rank candidate images by CLIP cosine similarity; return top-k rows."""
    q = idx.encode_text(visual_query)
    scores = embeddings @ q  # cosine, since both L2-normalized

    id_to_row = {iid: i for i, iid in enumerate(image_ids)}
    if candidate_ids is None:
        candidate_ids = image_ids
    cand_rows = [id_to_row[i] for i in candidate_ids if i in id_to_row]
    if not cand_rows:
        return []

    cand_scores = scores[cand_rows]
    order = np.argsort(-cand_scores)[:top_k]

    meta_by_id = df.set_index("image_id")
    results = []
    for rank, o in enumerate(order, start=1):
        row_idx = cand_rows[o]
        iid = image_ids[row_idx]
        meta = meta_by_id.loc[iid]
        results.append(
            {
                "rank": rank,
                "image_id": iid,
                "file_name": meta["file_name"],
                "image_path": meta["image_path"],
                "location": meta["location"],
                "date": str(meta["date"]),
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "similarity_score": round(float(cand_scores[o]), 4),
            }
        )
    return results


def explain_results_tool(
    query: str,
    parsed: dict,
    results: list[dict],
    n_candidates: int,
    enrich: bool = True,
) -> list[dict]:
    """Attach caption/tags (lazy vision) + a short explanation to each result."""
    cache = None
    if enrich:
        items = [(r["image_id"], ROOT / r["image_path"]) for r in results]
        cache = vision_extractor.describe_many(items)

    for r in results:
        desc = (cache or {}).get(r["image_id"], vision_extractor.FALLBACK)
        caption = desc.get("generated_caption", "No caption generated.")
        tags = (
            list(desc.get("objects", []))
            + list(desc.get("activities", []))
            + list(desc.get("scene", "").split() if desc.get("scene") not in ("", "unknown") else [])
        )
        tags = list(dict.fromkeys([t for t in tags if t]))  # dedup, keep order
        r["generated_caption"] = caption
        r["tags"] = tags

        matched = []
        if parsed.get("location"):
            matched.append(f"location={parsed['location']}")
        if parsed.get("date"):
            matched.append(f"date~{parsed['date']}")
        filt = f" matching {', '.join(matched)}" if matched else ""

        real_caption = caption != "No caption generated."
        if real_caption:
            expl = (
                f"Ranked {r['rank']} of {n_candidates} candidates{filt} by CLIP visual "
                f"similarity (score {r['similarity_score']}). Generated caption: \"{caption}\""
            )
            if tags:
                expl += f"; tags: {', '.join(tags[:6])}."
        else:
            expl = (
                f"Ranked {r['rank']} of {n_candidates} candidates{filt} by CLIP visual "
                f"similarity (score {r['similarity_score']}); caption/tags unavailable "
                f"(no OpenAI API key) — retrieval is pure CLIP image↔text matching."
            )
        r["explanation"] = expl
    return results
