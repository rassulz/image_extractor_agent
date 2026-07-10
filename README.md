# Image Extractor Agent — YDL 2026 Day 5

Notebook-first semantic image search. Images are embedded locally with **CLIP** (image encoder), natural-language queries with the CLIP text encoder, and retrieval is cosine similarity in the shared vector space. A vision model additionally generates captions/tags used for explanations and the Markdown/JSON report.

The project runs inside an **Anaconda (conda) environment**.

`captions.txt` from Flickr8k is deliberately excluded from the working dataset and is never used for indexing or retrieval.

## Quick start

Run in **Anaconda Prompt** (or PowerShell after a one-time `conda init powershell` + restart):

```bash
cd image_extractor_agent
conda create -n image-extractor python=3.11 -y
conda activate image-extractor
pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA build for NVIDIA GPU; skip if CPU-only
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY is optional — only for vision captions/explanations
```

The dataset script downloads Flickr8k via `kagglehub` (`adityajn105/flickr8k`), creates a deterministic 3,000-image subset and synthetic date/location labels, because Flickr8k does not include reliable capture metadata:

```bash
python scripts/prepare_flickr8k_subset.py --count 3000 --remove-raw
```

Then launch `jupyter notebook image_extractor_agent_day5.ipynb` from the activated env and run the cells top-to-bottom. Semantic search runs fully locally on CLIP embeddings (`ViT-B-32`/`laion2b_s34b_b79k`) and needs no API key; an `OPENAI_API_KEY` only adds generated captions/tags and richer explanations in the report. CLIP scores are low in absolute terms (~0.2–0.35 for good matches) — rank by them, don't threshold them.

## Project layout

- `data/images/` — 3,000-image Flickr8k subset (local only)
- `data/metadata.csv` — file-level metadata; location/date are explicitly synthetic
- `cache/` — vision descriptions and embedding cache
- `src/` — independently testable agent tools
- `outputs/` — latest JSON and Markdown search artefacts
