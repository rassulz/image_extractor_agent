"""Optional vision captions/tags via OpenAI. Lazy + cached.

Never enters the retrieval index — captions/tags are only for explanations
and the report. Falls back to documented placeholder records with no API key
or on any failure, so the pipeline never crashes.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cache" / "generated_image_descriptions.json"

FALLBACK = {
    "generated_caption": "No caption generated.",
    "objects": [],
    "scene": "unknown",
    "activities": [],
    "visual_attributes": [],
    "confidence_notes": "Vision extraction unavailable (no API key or failure).",
}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


_PROMPT = """You are a vision tagging tool. Look at the image and return ONLY a JSON object:
{
  "generated_caption": "one concise sentence describing the image",
  "objects": ["main", "objects"],
  "scene": "short scene description",
  "activities": ["actions"],
  "visual_attributes": ["colors", "lighting", "setting"],
  "confidence_notes": "one short sentence"
}"""


def describe_image(image_id: str, image_path: Path, cache: dict | None = None) -> dict:
    """Return a structured description dict for one image (cached, lazy)."""
    own_cache = cache is None
    cache = _load_cache() if own_cache else cache
    if image_id in cache:
        return cache[image_id]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        rec = {"image_id": image_id, **FALLBACK}
        cache[image_id] = rec
        if own_cache:
            _save_cache(cache)
        return rec

    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{_b64(image_path)}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        rec = {"image_id": image_id, **FALLBACK, **data}
    except Exception as e:  # noqa: BLE001
        rec = {"image_id": image_id, **FALLBACK}
        rec["confidence_notes"] = f"Vision extraction failed: {e}"

    cache[image_id] = rec
    if own_cache:
        _save_cache(cache)
    return rec


def describe_many(items: list[tuple[str, Path]]) -> dict:
    """Describe several (image_id, path) pairs, sharing one cache write."""
    cache = _load_cache()
    for image_id, path in items:
        describe_image(image_id, path, cache=cache)
    _save_cache(cache)
    return cache
