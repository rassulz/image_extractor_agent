"""Image Extractor Agent — orchestrates the search tools."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import dataset_loader, embedding_index, report_writer, search_tools

ROOT = Path(__file__).resolve().parents[1]


class ImageExtractorAgent:
    """Loads the metadata + CLIP index once, answers many queries."""

    def __init__(self, enrich: bool = True):
        self.df = dataset_loader.load_metadata()
        self.embeddings, self.image_ids = embedding_index.load_index(
            self.df["image_id"].tolist()
        )
        self.enrich = enrich

    def search(self, query: str, top_k: int = 5, save: bool = True,
               enrich: bool | None = None) -> dict:
        enrich = self.enrich if enrich is None else enrich
        parsed = search_tools.parse_query_tool(query, top_k=top_k)

        filters = {"date": parsed["date"]}
        has_filter = bool(filters["date"])
        candidate_ids = (
            search_tools.metadata_filter_tool(self.df, filters) if has_filter else None
        )

        if has_filter and not candidate_ids:
            answer = {
                "query": query,
                "interpreted_intent": parsed["interpreted_intent"],
                "filters": filters,
                "top_k": top_k,
                "results": [],
                "message": f"No images match filter(s): {filters}",
            }
            if save:
                paths = report_writer.save_report(answer)
                answer["report_path"] = paths["md"]
            return answer

        n_candidates = len(candidate_ids) if candidate_ids is not None else len(self.image_ids)
        results = search_tools.search_images_tool(
            parsed["visual_query"], self.df, self.embeddings, self.image_ids,
            candidate_ids=candidate_ids, top_k=top_k,
        )
        results = search_tools.explain_results_tool(
            query, parsed, results, n_candidates, enrich=enrich
        )

        answer = {
            "query": query,
            "interpreted_intent": parsed["interpreted_intent"],
            "filters": filters,
            "top_k": top_k,
            "results": results,
        }
        if save:
            paths = report_writer.save_report(answer)
            answer["report_path"] = paths["md"]
        return answer


def build_index_if_needed(batch_size: int = 64) -> None:
    """Build the CLIP index if the cache is missing or stale."""
    df = dataset_loader.load_metadata()
    ids = df["image_id"].tolist()
    if embedding_index.index_is_fresh(ids):
        print("[agent] CLIP index is fresh — reusing cache.")
        return
    paths = [ROOT / p for p in df["image_path"].tolist()]
    print(f"[agent] Building CLIP index for {len(ids)} images ...")
    embedding_index.build_index(ids, paths, batch_size=batch_size)
