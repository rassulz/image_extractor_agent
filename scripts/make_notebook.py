"""Generate image_extractor_agent_day5.ipynb from the src/ modules.

Keeping the notebook generated keeps it in sync with the tested code.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""# Image Extractor Agent — Day 5 (YDL 2026)

Natural-language semantic image search over a **3,000-image Flickr8k subset**.

**Pipeline (fully local, no API key needed for retrieval):**

```
Images + metadata → CLIP image encoder → image embeddings (the index)
User query        → CLIP text encoder  → cosine similarity → top-k images
```

- Embedding model: **CLIP `ViT-B-32` / `laion2b_s34b_b79k`** (open_clip).
- `captions.txt` is **never** used for indexing (see the rule below).
- Optional OpenAI vision captions/tags are used **only** for explanations/reports — never for retrieval.
- Runs inside the **conda** env `practical-ai-engineering`.
"""),
    md("## 1–2. Setup and imports"),
    code("""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src import dataset_loader, embedding_index, search_tools, report_writer
from src.agent import ImageExtractorAgent, build_index_if_needed
print("imports OK — CLIP model:", embedding_index.MODEL_TAG)
"""),
    md("""## 3–4. Prepare dataset

Run once in the terminal (downloads Flickr8k via kagglehub, builds the subset + `metadata.csv`):

```bash
python scripts/prepare_flickr8k_subset.py --count 3000 --remove-raw
```
`location`/`date` are **synthetic** (seeded RNG) — Flickr8k has no capture metadata."""),
    md("## 5. Load metadata"),
    code("""
df = dataset_loader.load_metadata()
print(f"{len(df)} images")
df.head()
"""),
    md("## 6. Display sample images"),
    code("""
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, (_, r) in zip(axes, df.sample(5, random_state=1).iterrows()):
    ax.imshow(Image.open(ROOT / r.image_path)); ax.axis("off")
    ax.set_title(f"{r.location}\\n{r.date}", fontsize=9)
plt.tight_layout(); plt.show()
"""),
    md("""## 7. Build the CLIP image-embedding index

Embeds every image with the CLIP **image** encoder (batched, `torch.no_grad()`),
L2-normalizes, and caches `cache/image_embeddings.npy` + `cache/embeddings_meta.json`.
Re-running reuses the cache unless it is stale (ids/model mismatch)."""),
    code("""
build_index_if_needed(batch_size=64)
emb, ids = embedding_index.load_index(df["image_id"].tolist())
print("index shape:", emb.shape)
"""),
    md("## 8. Semantic search — query → CLIP text encoder → cosine top-k"),
    code("""
def show_results(answer):
    res = answer["results"]
    print("Q:", answer["query"], "| intent:", answer["interpreted_intent"])
    if not res:
        print(answer.get("message", "no results")); return
    fig, axes = plt.subplots(1, len(res), figsize=(4*len(res), 4))
    if len(res) == 1: axes = [axes]
    for ax, r in zip(axes, res):
        ax.imshow(Image.open(ROOT / r["image_path"])); ax.axis("off")
        ax.set_title(f"#{r['rank']} {r['similarity_score']:.3f}\\n{r['location']} {r['date']}", fontsize=9)
    plt.tight_layout(); plt.show()

agent = ImageExtractorAgent(enrich=False)   # enrich=True adds OpenAI captions (needs key)
show_results(agent.search("a dog running on grass", top_k=5))
"""),
    md("## 9. Metadata filters — filter BEFORE ranking"),
    code("""
# "animals in Almaty" -> location=Almaty filter applied before top-k
show_results(agent.search("animals in Almaty", top_k=5))
"""),
    md("""## 10–11. Optional: OpenAI vision captions/tags (lazy) + searchable text

Only runs with `OPENAI_API_KEY`. Captions/tags are generated **lazily** for retrieved
images and cached in `cache/generated_image_descriptions.json`. They feed explanations
and the report **only** — never the retrieval index."""),
    code("""
# enrich=True captions just the top-k hits (a handful of API calls, cached).
enriched = agent.search("a person riding a bicycle", top_k=3, enrich=True)
for r in enriched["results"]:
    print(r["rank"], r["similarity_score"], "|", r["generated_caption"])
    print("   tags:", r["tags"])
"""),
    md("## 12–13. Agent tools + demo queries"),
    code("""
DEMO = [
    "a dog running outside",
    "people standing near water",
    "a child playing outdoors",
    "a person riding a bicycle",
    "several people in a city or street scene",
]
for q in DEMO:
    show_results(agent.search(q, top_k=5))
"""),
    md("## 14. Save JSON + Markdown reports"),
    code("""
ans = agent.search("a dog running on grass", top_k=5)
print("report saved to:", ans["report_path"])
print(Path(ROOT / "outputs" / "search_report.md").read_text()[:600])
"""),
    md("## 15. Mini evaluation"),
    code("""
rows = []
for q in DEMO:
    a = agent.search(q, top_k=1, save=False)
    top = a["results"][0] if a["results"] else {}
    rows.append({"query": q, "top_result_image_id": top.get("image_id"),
                 "score": top.get("similarity_score"), "manual_judgment": "?", "notes": ""})
pd.DataFrame(rows)
"""),
    md("""## 16. Demo script / architecture

I built an **Image Extractor Agent**: it searches an image dataset with natural language,
even though the dataset is treated as *images + basic metadata only*.

- `captions.txt` is intentionally **not** used. Every image is embedded locally with **CLIP**.
- The query is embedded with the CLIP **text** encoder and matched by cosine similarity — a single numpy matmul over the cached index, so search is instant.
- Metadata filters (location/date) are applied **before** ranking.
- An optional vision model adds captions/tags for explanations and the Markdown/JSON report.

A **React + Vite + Tailwind** front-end (`frontend/`) served by a **FastAPI** backend
(`backend/main.py`) provides the interactive demo, including an **Atlas** view that projects
the CLIP embedding space to 2D (PCA + KMeans) with auto-labeled semantic clusters.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "image_extractor_agent_day5.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
