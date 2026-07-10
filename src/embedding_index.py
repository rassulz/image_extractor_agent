"""CLIP image/text encoding + local vector index.

The retrieval index is PURE CLIP image embeddings. Text queries are encoded
with the CLIP text encoder into the same space; cosine similarity = matmul on
L2-normalized vectors. No captions or metadata ever enter the index.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
EMB_PATH = CACHE_DIR / "image_embeddings.npy"
META_PATH = CACHE_DIR / "embeddings_meta.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
MODEL_TAG = f"{MODEL_NAME}/{PRETRAINED}"

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _lazy_load():
    global _model, _preprocess, _tokenizer, _device
    if _model is not None:
        return
    import open_clip
    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    _model = _model.to(_device).eval()
    _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    print(f"[embedding_index] CLIP {MODEL_TAG} loaded on {_device}")


def encode_images(image_paths: list[Path], batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized image embeddings, one row per path."""
    import torch
    from PIL import Image

    _lazy_load()
    vecs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(_preprocess(Image.open(p).convert("RGB")))
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] could not read {p}: {e}; using zero vector")
                imgs.append(None)
        # Replace unreadable images with zero tensors after encoding.
        good_idx = [j for j, im in enumerate(imgs) if im is not None]
        batch = torch.stack([imgs[j] for j in good_idx]).to(_device) if good_idx else None
        out = np.zeros((len(batch_paths), _model.visual.output_dim), dtype=np.float32)
        if batch is not None:
            with torch.no_grad():
                feats = _model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out[good_idx] = feats.cpu().numpy().astype(np.float32)
        vecs.append(out)
        print(f"  encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
    return np.vstack(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)


def encode_text(query: str) -> np.ndarray:
    """Return a single L2-normalized text embedding for a query."""
    import torch

    _lazy_load()
    tokens = _tokenizer([query]).to(_device)
    with torch.no_grad():
        feat = _model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)[0]


def build_index(image_ids: list[str], image_paths: list[Path], batch_size: int = 64) -> None:
    """Encode all images and persist embeddings + sidecar meta."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb = encode_images(image_paths, batch_size=batch_size)
    np.save(EMB_PATH, emb)
    META_PATH.write_text(
        json.dumps({"model": MODEL_TAG, "image_ids": image_ids}, indent=2)
    )
    print(f"[embedding_index] saved {emb.shape} -> {EMB_PATH}")


def load_index(expected_ids: Iterable[str] | None = None):
    """Load embeddings + ordered image_ids. Validate against expected_ids/model.

    Returns (embeddings, image_ids). Raises if cache is stale or missing.
    """
    if not EMB_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index cache missing. Run build_index first.")
    meta = json.loads(META_PATH.read_text())
    if meta.get("model") != MODEL_TAG:
        raise ValueError(
            f"Stale index: cached model {meta.get('model')} != current {MODEL_TAG}"
        )
    image_ids = meta["image_ids"]
    if expected_ids is not None and list(expected_ids) != list(image_ids):
        raise ValueError("Stale index: image_ids differ from metadata.csv. Rebuild.")
    emb = np.load(EMB_PATH)
    if emb.shape[0] != len(image_ids):
        raise ValueError("Stale index: row count != number of image_ids.")
    return emb, image_ids


def index_is_fresh(expected_ids: list[str]) -> bool:
    try:
        load_index(expected_ids)
        return True
    except Exception:
        return False
