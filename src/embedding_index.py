"""Small local vector index with OpenAI embeddings and a TF-IDF fallback."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .vision_extractor import get_openai_client


@dataclass
class LocalEmbeddingIndex:
    """Embeddings plus the encoder required to turn a text query into a vector."""

    image_ids: list[str]
    embeddings: np.ndarray
    backend: str
    model: str
    vectorizer: TfidfVectorizer | None = None
    client: Any | None = None

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if self.backend == "tfidf":
            if self.vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer was not loaded.")
            return self.vectorizer.transform([query]).toarray().astype(np.float32)[0]
        if self.client is None:
            self.client = get_openai_client()
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is required to query an OpenAI embedding index.")
        response = self.client.embeddings.create(model=self.model, input=[query])
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    def similarity_scores(self, query: str) -> np.ndarray:
        query_embedding = self.embed_query(query)
        query_norm = np.linalg.norm(query_embedding)
        matrix_norms = np.linalg.norm(self.embeddings, axis=1)
        if query_norm == 0:
            return np.zeros(len(self.image_ids), dtype=np.float32)
        return (self.embeddings @ query_embedding) / np.maximum(matrix_norms * query_norm, 1e-12)


def _fingerprint(records: Iterable[dict[str, Any]]) -> str:
    stable_items = [
        {"image_id": str(record["image_id"]), "searchable_text": str(record["searchable_text"])}
        for record in records
    ]
    encoded = json.dumps(stable_items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_paths(cache_dir: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(cache_dir)
    return (
        directory / "image_embeddings.npy",
        directory / "image_embeddings.meta.json",
        directory / "tfidf_vectorizer.pkl",
    )


def _openai_embeddings(texts: list[str], *, client: Any, model: str, batch_size: int = 100) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=model, input=texts[start : start + batch_size])
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


def _load_cached_index(
    records: list[dict[str, Any]],
    *,
    cache_dir: str | Path,
    backend: str,
    model: str,
    client: Any | None,
) -> LocalEmbeddingIndex | None:
    embedding_path, meta_path, vectorizer_path = _cache_paths(cache_dir)
    if not embedding_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_ids = [str(record["image_id"]) for record in records]
        if (
            meta.get("fingerprint") != _fingerprint(records)
            or meta.get("backend") != backend
            or meta.get("model") != model
            or meta.get("image_ids") != expected_ids
        ):
            return None
        embeddings = np.load(embedding_path)
        if len(embeddings) != len(records):
            return None
        vectorizer = None
        if backend == "tfidf":
            if not vectorizer_path.is_file():
                return None
            with vectorizer_path.open("rb") as stream:
                vectorizer = pickle.load(stream)
        return LocalEmbeddingIndex(expected_ids, embeddings, backend, model, vectorizer, client)
    except (OSError, ValueError, pickle.UnpicklingError, json.JSONDecodeError):
        return None


def _save_cached_index(index: LocalEmbeddingIndex, records: list[dict[str, Any]], cache_dir: str | Path) -> None:
    embedding_path, meta_path, vectorizer_path = _cache_paths(cache_dir)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, index.embeddings)
    meta = {
        "fingerprint": _fingerprint(records),
        "backend": index.backend,
        "model": index.model,
        "image_ids": index.image_ids,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if index.backend == "tfidf" and index.vectorizer is not None:
        with vectorizer_path.open("wb") as stream:
            pickle.dump(index.vectorizer, stream)


def build_or_load_embedding_index(
    records: list[dict[str, Any]],
    *,
    cache_dir: str | Path = "cache",
    backend: str = "auto",
    model: str | None = None,
    client: Any | None = None,
) -> LocalEmbeddingIndex:
    """Build/reuse a local index. ``auto`` prefers OpenAI, otherwise TF-IDF."""
    if not records:
        raise ValueError("Cannot build a search index from zero records.")
    if backend not in {"auto", "openai", "tfidf"}:
        raise ValueError("backend must be 'auto', 'openai', or 'tfidf'.")

    client = client if client is not None else get_openai_client()
    chosen_backend = "openai" if backend == "openai" or (backend == "auto" and client is not None) else "tfidf"
    if chosen_backend == "openai" and client is None:
        raise RuntimeError("OPENAI_API_KEY is required when backend='openai'.")
    chosen_model = model or (
        os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") if chosen_backend == "openai" else "tfidf-v1"
    )
    cached = _load_cached_index(
        records, cache_dir=cache_dir, backend=chosen_backend, model=chosen_model, client=client
    )
    if cached is not None:
        return cached

    texts = [str(record["searchable_text"]) for record in records]
    image_ids = [str(record["image_id"]) for record in records]
    if chosen_backend == "openai":
        try:
            embeddings = _openai_embeddings(texts, client=client, model=chosen_model)
            index = LocalEmbeddingIndex(image_ids, embeddings, "openai", chosen_model, client=client)
        except Exception:
            if backend == "openai":
                raise
            chosen_backend, chosen_model = "tfidf", "tfidf-v1"
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            embeddings = vectorizer.fit_transform(texts).toarray().astype(np.float32)
            index = LocalEmbeddingIndex(image_ids, embeddings, chosen_backend, chosen_model, vectorizer)
    else:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        index = LocalEmbeddingIndex(image_ids, embeddings, chosen_backend, chosen_model, vectorizer)

    _save_cached_index(index, records, cache_dir)
    return index
