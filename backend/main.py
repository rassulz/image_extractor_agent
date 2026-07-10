"""FastAPI backend for the Image Extractor Agent React demo.

Endpoints:
  GET  /api/health            -> index status
  GET  /api/samples?n=8       -> random sample images
  POST /api/search            -> {query, top_k} -> agent answer
  GET  /images/<file_name>    -> static image bytes
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.agent import ImageExtractorAgent  # noqa: E402
from src import dataset_loader  # noqa: E402

app = FastAPI(title="Image Extractor Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR = ROOT / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

_agent: ImageExtractorAgent | None = None
_error: str | None = None


def get_agent() -> ImageExtractorAgent:
    global _agent, _error
    if _agent is None:
        try:
            # Pure-CLIP search by default: fast + zero API tokens.
            # OpenAI captions are opt-in per request (see /api/search).
            _agent = ImageExtractorAgent(enrich=False)
            _error = None
        except Exception as e:  # noqa: BLE001
            _error = str(e)
            raise HTTPException(status_code=503, detail=f"Index not ready: {e}")
    return _agent


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    enrich: bool = False  # opt-in OpenAI captions/tags (costs tokens)


@app.get("/api/health")
def health():
    try:
        agent = get_agent()
        return {
            "status": "ready",
            "num_images": len(agent.image_ids),
            "model": "CLIP ViT-B-32 / laion2b_s34b_b79k",
            "enrich_captions": agent.enrich,
        }
    except HTTPException:
        return {"status": "not_ready", "error": _error}


@app.get("/api/samples")
def samples(n: int = 8):
    df = dataset_loader.load_metadata()
    sample = df.sample(min(n, len(df)), random_state=7)
    return [
        {
            "image_id": r.image_id,
            "file_name": r.file_name,
            "url": f"/images/{r.file_name}",
            "location": r.location,
            "date": str(r.date),
            "width": int(r.width),
            "height": int(r.height),
        }
        for r in sample.itertuples()
    ]


_atlas_cache: dict | None = None

# Candidate concept labels for auto-naming clusters (label -> CLIP prompt).
_CONCEPTS = {
    "animals": "a photo of an animal or a dog",
    "water": "a photo of water, a lake, the sea or a beach",
    "city": "a photo of a city street or buildings",
    "sport": "a photo of people playing sport or running",
    "nature": "a photo of nature, grass, mountains or forest",
    "people": "a photo of a group of people",
    "children": "a photo of a child playing",
    "transport": "a photo of a bike, car or vehicle",
}
_CLUSTER_COLORS = ["#5DCAA5", "#7F9BE0", "#F0997B", "#AFA9EC", "#FAC775", "#8BC98B", "#F4A0C0", "#8FD0D8"]


@app.get("/api/atlas")
def atlas(k: int = 6):
    """2D semantic map of the whole index: PCA coords + KMeans clusters."""
    global _atlas_cache
    if _atlas_cache is not None:
        return _atlas_cache

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    from src import embedding_index as idx

    agent = get_agent()
    emb = agent.embeddings
    df = agent.df.set_index("image_id")

    coords = PCA(n_components=2, random_state=0).fit_transform(emb)
    mn, mx = coords.min(0), coords.max(0)
    coords = (coords - mn) / (mx - mn + 1e-9)

    k = max(2, min(k, len(_CLUSTER_COLORS)))
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(emb)
    labels = km.labels_

    # Auto-name each cluster by the concept most similar to its centroid.
    concept_names = list(_CONCEPTS.keys())
    concept_vecs = np.stack([idx.encode_text(p) for p in _CONCEPTS.values()])
    clusters = []
    used = set()
    for c in range(k):
        centroid = emb[labels == c].mean(0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = concept_vecs @ centroid
        order = np.argsort(-sims)
        name = next((concept_names[i] for i in order if concept_names[i] not in used), concept_names[order[0]])
        used.add(name)
        pts = coords[labels == c]
        clusters.append({
            "id": c,
            "label": name,
            "color": _CLUSTER_COLORS[c],
            "cx": round(float(pts[:, 0].mean()), 4),
            "cy": round(float(pts[:, 1].mean()), 4),
            "count": int((labels == c).sum()),
        })

    points = []
    ids = agent.image_ids
    for i, iid in enumerate(ids):
        row = df.loc[iid]
        points.append({
            "image_id": iid,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(labels[i]),
            "url": f"/images/{row['file_name']}",
        })

    _atlas_cache = {"points": points, "clusters": clusters, "total": len(points)}
    return _atlas_cache


@app.post("/api/search")
def search(req: SearchRequest):
    agent = get_agent()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    answer = agent.search(
        req.query, top_k=max(1, min(req.top_k, 20)), enrich=req.enrich
    )
    for r in answer.get("results", []):
        r["url"] = f"/images/{r['file_name']}"
    return answer
