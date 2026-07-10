"""Vision caption extraction with a safe local fallback and on-disk cache."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable


DESCRIPTION_KEYS = (
    "generated_caption",
    "objects",
    "scene",
    "activities",
    "visual_attributes",
    "confidence_notes",
)
VISION_PROMPT = """Analyze this image for semantic image search. Return JSON only, with exactly:
generated_caption (one precise sentence), objects (array of nouns), scene (short phrase),
activities (array), visual_attributes (array of visible properties), confidence_notes (short caveat).
Describe only visible content; do not guess location, date, names, or sensitive attributes."""


def fallback_description(reason: str = "Vision extraction is unavailable.") -> dict[str, Any]:
    return {
        "generated_caption": "No caption generated.",
        "objects": [],
        "scene": "unknown",
        "activities": [],
        "visual_attributes": [],
        "confidence_notes": reason,
    }


def get_openai_client(api_key: str | None = None) -> Any | None:
    """Create a client only when credentials and the optional SDK are available."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(api_key=api_key)


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalise_description(payload: dict[str, Any]) -> dict[str, Any]:
    description = fallback_description()
    for key in DESCRIPTION_KEYS:
        if key in payload and payload[key] is not None:
            description[key] = payload[key]
    for key in ("objects", "activities", "visual_attributes"):
        value = description[key]
        description[key] = [str(item).strip() for item in value] if isinstance(value, list) else []
    for key in ("generated_caption", "scene", "confidence_notes"):
        description[key] = str(description[key]).strip()
    return description


def describe_image_with_vision(
    image_path: str | Path,
    *,
    client: Any,
    model: str | None = None,
) -> dict[str, Any]:
    """Request a structured visual description through the OpenAI Responses API."""
    response = client.responses.create(
        model=model or os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_PROMPT},
                    {"type": "input_image", "image_url": _image_data_url(image_path)},
                ],
            }
        ],
    )
    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return _normalise_description(json.loads(text))


def load_description_cache(cache_path: str | Path = "cache/generated_image_descriptions.json") -> dict[str, dict[str, Any]]:
    path = Path(cache_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_description_cache(cache: dict[str, dict[str, Any]], cache_path: str | Path) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def enrich_records_with_vision(
    records: Iterable[dict[str, Any]],
    *,
    cache_path: str | Path = "cache/generated_image_descriptions.json",
    client: Any | None = None,
    model: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Enrich image records, reusing saved results and isolating per-image failures."""
    cache = load_description_cache(cache_path)
    client = client if client is not None else get_openai_client()
    enriched: list[dict[str, Any]] = []
    generated = 0
    for record in records:
        item = dict(record)
        image_id = str(item["image_id"])
        cached = cache.get(image_id)
        if cached:
            item.update(_normalise_description(cached))
        elif client is None:
            item.update(fallback_description("No OPENAI_API_KEY; image description was not generated."))
        elif limit is not None and generated >= limit:
            item.update(fallback_description("Skipped because the configured vision processing limit was reached."))
        else:
            try:
                item.update(describe_image_with_vision(item["image_path"], client=client, model=model))
                cache[image_id] = {key: item[key] for key in DESCRIPTION_KEYS}
                generated += 1
            except Exception as error:  # one bad image/API response must not stop indexing
                item.update(fallback_description(f"Vision extraction failed: {type(error).__name__}."))
        enriched.append(item)
    save_description_cache(cache, cache_path)
    return enriched
