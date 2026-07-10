"""Lightweight orchestration layer for the Image Extractor Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dataset_loader import build_searchable_text, load_metadata, records_from_metadata
from .embedding_index import LocalEmbeddingIndex, build_or_load_embedding_index
from .report_writer import save_report_tool, save_results_json
from .search_tools import explain_results_tool, metadata_filter_tool, parse_query_tool, search_images_tool
from .vision_extractor import enrich_records_with_vision, get_openai_client


@dataclass
class ImageExtractorAgent:
    records: list[dict[str, Any]]
    index: LocalEmbeddingIndex

    @classmethod
    def from_local_dataset(
        cls,
        *,
        metadata_path: str = "data/metadata.csv",
        cache_dir: str = "cache",
        embedding_backend: str = "auto",
        vision_limit: int | None = None,
    ) -> "ImageExtractorAgent":
        metadata = load_metadata(metadata_path)
        records = records_from_metadata(metadata)
        records = enrich_records_with_vision(
            records,
            cache_path=f"{cache_dir}/generated_image_descriptions.json",
            client=get_openai_client(),
            limit=vision_limit,
        )
        for record in records:
            record["searchable_text"] = build_searchable_text(record)
        index = build_or_load_embedding_index(records, cache_dir=cache_dir, backend=embedding_backend)
        return cls(records=records, index=index)

    def run(self, user_query: str, top_k: int = 5) -> dict[str, Any]:
        known_locations = sorted({str(record["location"]) for record in self.records if record.get("location")})
        parsed = parse_query_tool(user_query, known_locations)
        # Retrieve extra candidates before filtering so a selective city/date still gets top_k results.
        candidates = search_images_tool(self.index, self.records, parsed["visual_query"], top_k=None)
        filtered = metadata_filter_tool(candidates, parsed)
        results = explain_results_tool(user_query, filtered[:top_k])
        report_path = save_report_tool(user_query, parsed["visual_query"], parsed, results)
        payload = {
            "query": user_query,
            "interpreted_intent": parsed["visual_query"],
            "filters": {"location": parsed.get("location"), "date": parsed.get("date")},
            "top_k": top_k,
            "results": results,
            "report_path": report_path,
        }
        payload["results_json_path"] = save_results_json(payload)
        return payload


def run_image_extractor_agent(user_query: str, top_k: int = 5, **setup: Any) -> dict[str, Any]:
    """Convenience notebook entry point matching the Day 5 demo flow."""
    agent = ImageExtractorAgent.from_local_dataset(**setup)
    return agent.run(user_query, top_k=top_k)
