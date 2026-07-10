# Image Extractor Agent

**Natural-language semantic image search** — type *"a dog running on grass"* and instantly get the matching photos from a 3,000-image [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) subset, ranked by how well they *visually* match your words.

No captions, no tags, no text metadata are used for retrieval — the system understands the images themselves. Images are embedded locally with the **CLIP** image encoder, the query with the CLIP **text** encoder, and retrieval is cosine similarity in their shared vector space: one numpy matmul over a cached index, so **search is instant and works fully offline**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CLIP%20ViT--B%2F32-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React%20%2B%20Vite%20%2B%20Tailwind-frontend-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

## 🎬 Video demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/VJkXTofwaTo/maxresdefault.jpg)](https://youtu.be/VJkXTofwaTo)

▶ **[Watch the full walkthrough on YouTube](https://youtu.be/VJkXTofwaTo)** — semantic search, the 2D embedding Atlas, and the agent pipeline in action.

## Features

- 🔍 **Semantic search (Поиск)** — describe a scene in natural language, get ranked photos with a match %, metadata, and an explanation of *why* each image matched.
- 🗺️ **Semantic Atlas (Атлас)** — the entire 512-D CLIP index projected to 2D with PCA, clustered with KMeans, and each cluster **auto-labeled with CLIP itself** (zero-shot against candidate label prompts). Hover and click to explore how the model organizes the dataset.
- 🤖 **Agent-style pipeline** — the query is parsed into visual intent + metadata filters (location / date), filters are applied **before** ranking, results are explained, and a Markdown/JSON report is saved.
- ⚡ **Offline-first** — CLIP runs locally (GPU if available); the whole retrieval path makes **zero API calls**. An `OPENAI_API_KEY` is optional and only adds generated captions/tags and richer explanations.
- ✅ **Honest retrieval** — Flickr8k's `captions.txt` is **never** used for indexing, retrieval, or ranking. The point of the project is that the system understands images directly.

## How it works

```text
                       offline (once)                          per query (instant)
┌────────────────┐   ┌──────────────────────┐    ┌──────────────────────────────────────┐
│ data/images/   │──►│ CLIP image encoder    │    │ "people near water in Astana"        │
│ 3,000 photos   │   │ (ViT-B/32, batched,   │    │        │                             │
│ + metadata.csv │   │  L2-normalized)       │    │        ▼                             │
└────────────────┘   └──────────┬───────────┘    │ parse → visual query + filters       │
                                ▼                │        │                             │
                     cache/image_embeddings.npy  │        ▼                             │
                     cache/embeddings_meta.json  │ metadata filter (before ranking)     │
                                │                │        │                             │
                                ▼                │        ▼                             │
                     ┌──────────────────────┐    │ CLIP text encoder → cosine top-k     │
                     │ local vector index    │◄──┤        │                             │
                     └──────────────────────┘    │        ▼                             │
                                                 │ explanations + Markdown/JSON report  │
                                                 └──────────────────────────────────────┘
```

- **Model:** `open_clip` ViT-B-32 (`laion2b_s34b_b79k`) — embeds 3,000 images in ~1–2 minutes on a 4 GB GPU.
- **Index:** pure CLIP image embeddings. Captions/metadata never enter the index vectors.
- **Vision enrichment (optional):** retrieved images can be captioned/tagged lazily by a vision model — used only for explanations and reports, never for retrieval.

## Quick start

```bash
git clone https://github.com/rassulz/image_extractor_agent.git
cd image_extractor_agent

conda activate practical-ai-engineering    # or your own env: conda create -n image-extractor python=3.11
pip install -r requirements.txt            # torch, open-clip, fastapi, …

# 1. Dataset — either unzip the bundled subset:
unzip images.zip -d data/                  # -> data/images/ (3,000 images, ~65 MB)
#    …or rebuild it deterministically from Kaggle:
# python scripts/prepare_flickr8k_subset.py --count 3000 --remove-raw

# 2. Frontend deps
cd frontend && npm install && cd ..

# 3. One command: build the CLIP index (once) + start backend + frontend
./run_demo.sh                              # CONDA_ENV=<name> ./run_demo.sh to override the env
# → open http://localhost:5173
```

`OPENAI_API_KEY` in `.env` is **optional** (see `.env.example`): it only enables generated captions/tags and LLM explanations. The backend does **pure-CLIP** search by default, so a demo costs zero API tokens.

### Manual run (without the script)

```bash
python scripts/build_index.py            # build/refresh the CLIP index
uvicorn backend.main:app --port 8000     # backend  → http://localhost:8000
cd frontend && npm run dev               # frontend → http://localhost:5173
```

### Notebook

The course-required notebook demo is [image_extractor_agent_day5.ipynb](image_extractor_agent_day5.ipynb) — the same pipeline end-to-end (dataset → embeddings → search → filters → agent → report), regenerable with `python scripts/make_notebook.py`.

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/api/search` | POST | Natural-language query → parsed intent, filters, top-k ranked images with scores and explanations |
| `/api/atlas?k=6` | GET | 2D PCA projection of the whole index + KMeans clusters auto-labeled with CLIP |
| `/api/samples?n=8` | GET | Random sample images for the landing gallery |
| `/api/health` | GET | Index status, model name, image count |
| `/images/…` | GET | Static dataset images |

## Project layout

```text
image_extractor_agent/
├── image_extractor_agent_day5.ipynb   # notebook demo (required deliverable)
├── run_demo.sh                        # build index + start backend + frontend
├── requirements.txt / .env.example
├── scripts/    prepare_flickr8k_subset.py · build_index.py · make_notebook.py
├── src/        dataset_loader · embedding_index · search_tools · vision_extractor · report_writer · agent
├── backend/    main.py                # FastAPI
├── frontend/   src/App.jsx · Atlas.jsx  # React + Vite + Tailwind
├── report/     Image_Extractor_Agent_Report.pdf
├── data/       images/ · metadata.csv   # local only (gitignored)
├── cache/      image_embeddings.npy · embeddings_meta.json · generated_image_descriptions.json
└── outputs/    search_results.json · search_report.md
```

## Design notes & limitations

- **CLIP cosine scores are low in absolute terms** (~0.2–0.32 for good cross-modal matches). They are meaningful for *ranking*, not thresholding — the UI maps them to a friendly match %.
- **Synthetic metadata:** Flickr8k has no real capture metadata, so `location` is `unknown` (none is invented) and `date` is generated from a seeded RNG — purely to demonstrate filter-before-ranking metadata filtering.
- **Stale-cache protection:** the index is validated against `metadata.csv` (ids + order + model name) on load; a mismatched cache is rejected and rebuilt instead of silently returning wrong images.
- **Deterministic dataset:** the 3,000-image subset is selected with sorted filenames + a fixed seed, so anyone can reproduce the exact same index.

## License

[Apache 2.0](LICENSE)
