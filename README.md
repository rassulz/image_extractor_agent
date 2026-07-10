# Image Extractor Agent — YDL 2026 Day 5

Notebook-first semantic image search. It builds visual descriptions from the images themselves, embeds the descriptions plus metadata, retrieves matching images, and saves a Markdown/JSON report.

`captions.txt` from Flickr8k is deliberately excluded from the working dataset and is never used for indexing or retrieval.

## Quick start

```bash
cd image_extractor_agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY for vision + semantic embeddings
```

The dataset script creates a deterministic subset and synthetic date/location labels, because Flickr8k does not include reliable capture metadata:

```bash
.venv/bin/python scripts/prepare_flickr8k_subset.py --count 3000 --remove-raw
```

Then open `image_extractor_agent_day5.ipynb` and run the cells top-to-bottom. Without an API key the search still runs with TF-IDF, but captions remain placeholders; use an API key to generate actual image descriptions.

## Project layout

- `data/images/` — 3,000-image Flickr8k subset (local only)
- `data/metadata.csv` — file-level metadata; location/date are explicitly synthetic
- `cache/` — vision descriptions and embedding cache
- `src/` — independently testable agent tools
- `outputs/` — latest JSON and Markdown search artefacts
