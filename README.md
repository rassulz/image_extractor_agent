# Image Extractor Agent — YDL 2026 Day 5

Natural-language **semantic image search** over a 3,000-image Flickr8k subset, with a
**React + Vite + Tailwind** front-end and a **FastAPI** backend.

Images are embedded locally with **CLIP** (image encoder); the user's natural-language
query is embedded with the CLIP **text** encoder; retrieval is cosine similarity in the
shared vector space — one numpy matmul over a cached index, so **search is instant**.
An optional vision model adds captions/tags for explanations and reports.

> `captions.txt` from Flickr8k is **never** used for indexing or retrieval — the point is
> that the system understands the images directly.

## What you get

- **`frontend/`** — React + Vite + Tailwind demo (pastel UI matching the provided mockups):
  - **Поиск** — type a description, get ranked photos with a match %, metadata and an explanation.
  - **Атлас** — the whole index projected to 2D (PCA) with auto-labeled semantic clusters (KMeans + CLIP).
- **`backend/main.py`** — FastAPI: `/api/search`, `/api/atlas`, `/api/samples`, `/api/health`, static images.
- **`src/`** — independently testable agent tools (dataset loader, CLIP index, search tools, vision extractor, report writer, agent).
- **`image_extractor_agent_day5.ipynb`** — the required notebook demo (regenerate with `python scripts/make_notebook.py`).

## Quick start (conda)

```bash
cd image_extractor_agent
conda activate practical-ai-engineering        # env used in Week 4
pip install -r requirements.txt                # torch, open-clip, fastapi, …

# 1. Dataset — either unzip the bundled compressed subset:
unzip images.zip -d data/            # -> data/images/ (3,000 images, ~65 MB)
#    or rebuild it from scratch (downloads Flickr8k via kagglehub):
# python scripts/prepare_flickr8k_subset.py --count 3000 --remove-raw

# 2. Front-end deps
cd frontend && npm install && cd ..

# 3. One command: build the CLIP index (once) + start backend + frontend
./run_demo.sh
# → open http://localhost:5173
```

`OPENAI_API_KEY` is **optional** (in `.env`): it only adds generated captions/tags and richer
explanations. Retrieval works fully offline without it. To keep the shared key's token use low,
the backend does **pure-CLIP** search by default (zero API calls per query).

## Manual run (without the script)

```bash
conda run -n practical-ai-engineering python scripts/build_index.py          # build/refresh index
conda run -n practical-ai-engineering uvicorn backend.main:app --port 8000   # backend
cd frontend && npm run dev                                                   # frontend :5173
```

## Notes

- CLIP cross-modal cosine is low in absolute terms (**~0.2–0.32** for good matches); rank by it, don't threshold it. The UI maps it to a friendly match %.
- `location` in `metadata.csv` is **unknown** — Flickr8k is an international dataset with no real location metadata, so none is invented. Only `date` is **synthetic** (seeded RNG), purely to demonstrate metadata filtering.
- The index is validated against `metadata.csv` (ids + order + model) on load, so a stale cache is rejected instead of silently returning wrong images.

## Project layout

```
image_extractor_agent/
├── image_extractor_agent_day5.ipynb   # required notebook demo
├── run_demo.sh                        # build index + start backend + frontend
├── requirements.txt / .env.example
├── scripts/  prepare_flickr8k_subset.py · build_index.py · make_notebook.py
├── src/      dataset_loader · embedding_index · search_tools · vision_extractor · report_writer · agent
├── backend/  main.py                  # FastAPI
├── frontend/ src/App.jsx · Atlas.jsx  # React + Vite + Tailwind
├── data/     images/ · metadata.csv   # local only (gitignored)
├── cache/    image_embeddings.npy · embeddings_meta.json · generated_image_descriptions.json
└── outputs/  search_results.json · search_report.md
```
