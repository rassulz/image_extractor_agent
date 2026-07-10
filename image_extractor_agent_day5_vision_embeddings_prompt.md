# Image Extractor Agent — Day 5 Capstone Implementation Prompt

## Context

We are building a **Day 5 Capstone project** for an AI Engineering course.

The project is an **Image Extractor Agent**.

The goal is to search a local image dataset using natural language, even when the original dataset contains only:

- image files
- basic metadata

The key pipeline is:

```text
Required, fully local (no API key):
Images + metadata
→ CLIP image encoder → image embeddings (the search index)
→ User query → CLIP text encoder → cosine similarity → top-k images

Optional enrichment (needs OPENAI_API_KEY):
Retrieved images → Vision caption/tag extraction
→ Searchable text → explanations + report content
```

The project must be implemented as a working notebook/demo, not as a production app.

Three hard requirements:

- The project must run inside an **Anaconda (conda) environment**, not a plain `venv`.
- The embedding model must be **CLIP**: images are embedded with the CLIP image encoder, text queries with the CLIP text encoder, so retrieval works fully locally without API calls. Default checkpoint: `open_clip` `ViT-B-32` / `laion2b_s34b_b79k` (~600MB, fits a 4GB-VRAM GPU; `transformers` `openai/clip-vit-base-patch32` is an equivalent alternative).
- The dataset is a **deterministic 3,000-image subset of Flickr8k**, downloaded with `kagglehub.dataset_download("adityajn105/flickr8k")` and prepared by `scripts/prepare_flickr8k_subset.py`.

---

## Project Name

**Image Extractor Agent**

Optional display names:

- VisionSearch Agent
- ImageSense Agent
- Semantic Image Finder
- Visual Memory Search

Use **Image Extractor Agent** as the main name unless there is a strong reason to rename it.

---

## Core Idea

The system receives a natural language query such as:

```text
Find images with a dog running on grass.
```

or:

```text
Show me outdoor images with people near water.
```

The agent should search a local dataset of images and return the most relevant image matches.

For each result, the system should show:

- image id
- file name
- image path
- location
- date
- width
- height
- generated caption
- tags
- similarity score
- short explanation of why the image matches the query

---

# Very Important Rule: Do NOT Use captions.txt for Indexing

If the dataset is Flickr8k or another dataset that contains `captions.txt`, do **not** use `captions.txt` as the input for building the search index.

The system must assume that the dataset contains only:

```text
image_id
file_name
image_path
location
date
width
height
```

The project should generate its own captions and tags from the images using a vision model.

Allowed use of `captions.txt`:

```text
Optional evaluation only
```

Not allowed:

```text
Do not use captions.txt as searchable text.
Do not embed captions.txt.
Do not use captions.txt to generate retrieval results.
Do not use captions.txt to cheat the search pipeline.
```

The point of the project is to demonstrate that the system can understand images directly.

---

# Target Architecture

```text
                    ┌──────────────────────┐
                    │ User Text Query       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Extractor Agent │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ Query Parser    │  │ Image Search Tool│  │ Metadata Filter   │
│ LLM / rules     │  │ Embedding search │  │ date/location     │
└─────────────────┘  └──────────────────┘  └───────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Explanation Tool      │
                    │ LLM or rule-based     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Markdown/JSON Report  │
                    └──────────────────────┘
```

Offline indexing pipeline:

```text
Images folder + metadata.csv
     │
     ├─► REQUIRED, local, no API key:
     │      CLIP image encoder (batched, GPU if available)
     │      → cache/image_embeddings.npy + cache/embeddings_meta.json
     │      → local vector index
     │
     └─► OPTIONAL, needs OPENAI_API_KEY (lazy — only for retrieved images):
            Vision extractor
              ├── generated caption
              ├── object tags
              ├── scene tags
              ├── activity tags
              └── visual attributes
            → searchable text builder
            → explanations + Markdown/JSON report
              (never enters the retrieval index)
```

---

# Recommended Tech Stack

Use Python notebook first.

The project must run in an **Anaconda (conda) environment**. Run these commands in **Anaconda Prompt**, or in PowerShell after a one-time `conda init powershell` and a shell restart:

```bash
conda create -n image-extractor python=3.11 -y
conda activate image-extractor
pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA build for NVIDIA GPUs (e.g. RTX 3050); no GPU → skip this line
pip install -r requirements.txt
```

Required:

```text
python (Anaconda / conda environment)
pandas
numpy
Pillow
matplotlib
scikit-learn
torch
open-clip-torch
kagglehub
python-dotenv
jupyter
```

Optional:

```text
openai  (only for caption/tag generation and explanations, not for retrieval)
transformers  (only if you prefer HF CLIPModel over open-clip-torch)
gradio
faiss-cpu
chromadb
```

Embeddings must come from **CLIP**, not from an embeddings API. Default model: `open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')` — fits a 4GB-VRAM GPU comfortably and embeds 3,000 images in ~1–2 minutes at batch size 64.

For Day 5, keep it simple. Prefer:

```text
Local JSON/CSV index + numpy cosine similarity
```

over a full vector database.

---

# Expected Project Structure

Create or organize the project like this:

```text
image-extractor-agent/
│
├── image_extractor_agent_day5.ipynb
├── README.md
├── requirements.txt
├── .env.example
│
├── scripts/
│   └── prepare_flickr8k_subset.py
│
├── data/
│   ├── raw/                  (temporary kagglehub download; removed by --remove-raw)
│   ├── images/               (the 3,000-image subset)
│   └── metadata.csv
│
├── cache/
│   ├── generated_image_descriptions.json
│   ├── image_embeddings.npy
│   └── embeddings_meta.json  (CLIP model name + ordered image_ids for the .npy rows)
│
├── outputs/
│   ├── search_results.json
│   └── search_report.md
│
└── src/
    ├── dataset_loader.py
    ├── vision_extractor.py
    ├── embedding_index.py
    ├── search_tools.py
    ├── report_writer.py
    └── agent.py
```

For the Day 5 demo, it is acceptable if most code lives inside the notebook.

However, the notebook should still be organized with clear sections.

---

# Dataset Requirements

## Input Dataset

The working dataset is fixed: a **deterministic 3,000-image subset of Flickr8k**.

Download the full dataset (~8,000 images) with kagglehub:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("adityajn105/flickr8k")

print("Path to dataset files:", path)
```

Then build the subset with `scripts/prepare_flickr8k_subset.py`:

```bash
python scripts/prepare_flickr8k_subset.py --count 3000 --remove-raw
```

The script must:

- download Flickr8k via kagglehub (or reuse an existing download);
- select 3,000 images deterministically (sorted filenames + fixed random seed);
- copy them into `data/images/` and never copy `captions.txt` there;
- generate `data/metadata.csv` programmatically: real width/height via PIL, synthetic location/date from a seeded RNG;
- sort `metadata.csv` by `image_id` so ordering is deterministic;
- optionally delete the raw download (`--remove-raw`).

While developing, iterate on the first ~50 images of the same subset, then build the full 3,000-image index once — CLIP makes this cheap (minutes, no API calls).

---

## Required Metadata File

Create `data/metadata.csv`.

Required columns:

```csv
image_id,file_name,image_path,location,date,width,height,source
```

Use the original Flickr8k filename stem as `image_id`. Example:

```csv
1000268201_693b08cb0e,1000268201_693b08cb0e.jpg,data/images/1000268201_693b08cb0e.jpg,Almaty,2026-07-01,1024,768,flickr8k
1001773457_577c3a7d70,1001773457_577c3a7d70.jpg,data/images/1001773457_577c3a7d70.jpg,Astana,2026-07-02,800,600,flickr8k
1002674143_1b742ab4b8,1002674143_1b742ab4b8.jpg,data/images/1002674143_1b742ab4b8.jpg,Atyrau,2026-07-03,1280,720,flickr8k
```

(Examples elsewhere in this document use short ids like `img_014` for readability only.)

`metadata.csv` is generated by `scripts/prepare_flickr8k_subset.py`, not written by hand: real width/height come from PIL, and location/date are synthetic (seeded RNG), because Flickr8k has no reliable capture metadata.

Example synthetic locations:

```text
Almaty
Astana
Atyrau
Shymkent
Aktau
Karaganda
```

Example synthetic dates:

```text
2026-07-01
2026-07-02
2026-07-03
...
```

The project should clearly say that location/date are synthetic if they are generated.

---

# Generated Image Description Schema

For each image, generate a structured description like this:

```json
{
  "image_id": "img_001",
  "file_name": "img_001.jpg",
  "generated_caption": "A dog is running across a grassy field outdoors.",
  "objects": ["dog", "grass", "field"],
  "scene": "outdoor field",
  "activities": ["running"],
  "visual_attributes": ["daylight", "green grass", "open area"],
  "confidence_notes": "The main visible subject appears to be a dog running outside."
}
```

The vision extractor should produce a similar structure.

If the vision model fails, the system should not crash. Use fallback values:

```json
{
  "generated_caption": "No caption generated.",
  "objects": [],
  "scene": "unknown",
  "activities": [],
  "visual_attributes": [],
  "confidence_notes": "Vision extraction failed for this image."
}
```

---

# Searchable Text Format

For each image, build one text field that combines generated visual information and metadata.

Example:

```text
Image ID: img_001.
Caption: A dog is running across a grassy field outdoors.
Objects: dog, grass, field.
Scene: outdoor field.
Activities: running.
Visual attributes: daylight, green grass, open area.
Location: Almaty.
Date: 2026-07-01.
Size: 1024x768.
```

The primary search index is built from **CLIP image embeddings**. The `searchable_text` is used for explanations and the report only — it is never embedded into the retrieval index. (CLIP's text encoder truncates at 77 BPE tokens; the example above already exceeds that, so it could not be embedded faithfully anyway.)

Do not embed raw metadata only.

Do not embed `captions.txt`.

---

# Embedding and Search

Use **CLIP** as the embedding model. Default: `open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')`; `transformers` `CLIPModel` (`openai/clip-vit-base-patch32`) is an equivalent alternative. CLIP is a dual encoder: the image encoder and the text encoder map into the same vector space, so a text query can be compared directly against image vectors.

## Embedding Step

For each image:

```text
image file → CLIP image encoder → image embedding vector
```

- Encode in batches (e.g. 64) under `torch.no_grad()`, on the GPU if available — 3,000 images take ~1–2 minutes on an RTX 3050, and the index is complete with **zero API calls**.
- L2-normalize the vectors before saving; cosine similarity then becomes a single numpy matmul.
- The index is **pure CLIP image embeddings**. Do not mix caption or metadata text into the index vectors: the synthetic location/date strings have no visual grounding in CLIP space and only add noise. Metadata influences results through the metadata filter, not through embeddings.

Save embeddings to:

```text
cache/image_embeddings.npy
```

Alongside the array, save a sidecar that ties rows to images and pins the model:

```text
cache/embeddings_meta.json
{
  "model": "ViT-B-32/laion2b_s34b_b79k",
  "image_ids": ["...ordered exactly like the .npy rows..."]
}
```

Save generated captions/tags (when available) to:

```text
cache/generated_image_descriptions.json
```

## Query Step

For user query:

```text
parsed visual query → CLIP text encoder → query embedding
```

Then calculate cosine similarity:

```text
similarity(query_embedding, image_embedding)
```

Return top-k results over the filtered candidate set (see Metadata Filtering — filters are applied **before** ranking).

Because CLIP runs locally, retrieval must work without any API key.

Recommended default:

```text
top_k = 5
```

Score expectations: CLIP image–text cosine similarity is low in absolute terms — a good match is typically **0.2–0.35** for ViT-B/32; 0.9+ never occurs across modalities. Interpret scores relatively for ranking; never compare them against a high absolute threshold.

---

# Metadata Filtering

Support basic metadata filtering if the user query includes filters.

Minimum filters:

```text
location
date
```

Optional filters:

```text
width
height
orientation
source
```

Examples:

```text
Find images with dogs in Almaty.
Find outdoor photos from July 2026.
Find images with people near water in Astana.
```

For Day 5, filters can be simple and rule-based.

Example approach:

```python
if "Almaty" in query:
    location_filter = "Almaty"
```

Better approach:

Use an LLM to parse the query into structured filters.

Example parsed query:

```json
{
  "visual_query": "dogs playing outside",
  "location": "Almaty",
  "date": null,
  "top_k": 5
}
```

## Filter before ranking (important)

Apply metadata filters **before** taking top-k, not after:

1. Build a boolean mask over `metadata.csv` from the filters (candidate set).
2. Rank only the masked rows of the embedding matrix by CLIP similarity (or score all 3,000 rows — one cheap matmul — and mask the score vector).
3. Take top-k from the survivors.

Filtering after top-k retrieval starves results: with ~6 synthetic locations, the global top-5 for "animals in Almaty" will usually contain zero Almaty images, even though hundreds of matching Almaty images exist in the index. With filter-then-rank, results can be empty only when the filter itself matches zero images — in that case the agent should say "no images match location=X" instead of returning a silent empty list.

---

# Agent Requirements

The agent should behave like an orchestrator.

It should not manually perform all logic inside one giant function.

Create small tools/functions:

```python
def parse_query_tool(query: str) -> dict:
    ...

def metadata_filter_tool(filters: dict) -> list:
    """Returns candidate image_ids (a mask over metadata.csv)."""
    ...

def search_images_tool(visual_query: str, candidate_ids: list | None = None, top_k: int = 5) -> list:
    """Ranks the candidate set (or all images) by CLIP similarity."""
    ...

def explain_results_tool(query: str, results: list) -> list:
    ...

def save_report_tool(query: str, results: list) -> str:
    ...
```

The agent flow:

```text
1. Receive user query
2. Parse query into visual intent and metadata filters
3. Apply metadata filters to select the candidate set
4. Rank candidates by CLIP similarity and take top-k
5. Explain each result
6. Save report
7. Return final structured answer
```

Explanations without an API key: build the rule-based fallback from what is always available — the CLIP similarity score, which metadata filters matched, and the query terms (e.g. "Ranked 1 of 512 Almaty candidates by CLIP visual similarity, score 0.31; caption/tags unavailable — no API key"). Mention caption/tag overlap only when a real (non-fallback) description exists.

---

# Agent Output Schema

The final output should look like this:

```json
{
  "query": "Find images with a dog running on grass",
  "interpreted_intent": "Search for images containing a dog running outdoors on grass",
  "filters": {
    "location": null,
    "date": null
  },
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "image_id": "img_014",
      "file_name": "img_014.jpg",
      "image_path": "data/images/img_014.jpg",
      "location": "Almaty",
      "date": "2026-07-01",
      "width": 1024,
      "height": 768,
      "similarity_score": 0.31,
      "generated_caption": "A brown dog runs through a grassy field.",
      "tags": ["dog", "grass", "field", "running"],
      "explanation": "This image matches because the generated caption and tags contain a dog, outdoor grass, and running activity."
    }
  ],
  "report_path": "outputs/search_report.md"
}
```

---

# Markdown Report Format

Save every search result to:

```text
outputs/search_report.md
```

The report should include:

```md
# Image Extractor Agent — Search Report

## User Query

Find images with a dog running on grass.

## Interpreted Intent

Search for images containing a dog running outdoors on grass.

## Filters

- Location: None
- Date: None

## Top Results

### 1. img_014.jpg

- Path: `data/images/img_014.jpg`
- Location: Almaty
- Date: 2026-07-01
- Size: 1024x768
- Similarity Score: 0.31
- Generated Caption: A brown dog runs through a grassy field.
- Tags: dog, grass, field, running
- Explanation: This image matches because the generated caption and tags contain a dog, outdoor grass, and running activity.

### 2. img_022.jpg

...
```

---

# Notebook Requirements

The main notebook should be:

```text
image_extractor_agent_day5.ipynb
```

Required sections (note: the CLIP index is built BEFORE any captioning, so the notebook works end-to-end without an API key):

```text
1. Project overview
2. Setup and imports
3. Load environment variables
4. Prepare dataset (kagglehub download + 3,000-image subset via scripts/prepare_flickr8k_subset.py)
5. Load image dataset and metadata.csv
6. Display sample images
7. Generate CLIP image embeddings (batched) and build the local vector index
8. Implement semantic search (query → CLIP text encoder → cosine top-k)
9. Implement metadata filters (filter before ranking)
10. Generate image captions and tags with vision model (optional, lazy — retrieved images only)
11. Build searchable text (for explanations and the report only)
12. Implement agent tools
13. Run agent demo queries
14. Save JSON and Markdown reports
15. Mini evaluation
16. Demo script / explanation
```

---

# Demo Queries

Use these queries for the final demo:

```text
Find images with a dog running outside.
Find images with people standing near water.
Find outdoor images with a child playing.
Find images with a bicycle or a person riding a bike.
Find images with several people in a city or street scene.
Find images taken in Almaty with animals.
Find images from July 2026 that show outdoor activity.
```

Make sure at least 3–5 demo queries return good results.

---

# Mini Evaluation

Add a small evaluation section with 3–5 test queries.

For each query, manually inspect whether the top result is reasonable.

Create a table:

```text
query
top_result_image_id
score
manual_judgment
notes
```

Example:

```csv
query,top_result_image_id,score,manual_judgment,notes
dog running on grass,img_014,0.31,good,Dog and grass clearly visible
people near water,img_022,0.29,good,People standing near lake
bicycle on street,img_031,0.27,partial,Bicycle present but not clearly on street
```

This is enough for Day 5.

---

# Error Handling

Implement basic error handling.

The system should handle:

- missing image file
- corrupted image
- missing metadata
- failed vision extraction
- failed embedding generation
- empty search results
- invalid user query

If an image fails during captioning, skip it or store fallback fields.

Do not let one failed image stop the whole notebook.

---

# Caching Requirement

Vision extraction costs API credits and wall-clock time; CLIP embedding of 3,000 images costs a couple of minutes. Cache both.

**Captions are lazy.** Do not caption all 3,000 images up front — sequential vision calls would take roughly 2–4 hours. After a query returns top-k, caption only the uncached hit images and append them to the cache keyed by `image_id`; the cache accumulates across queries (tens of API calls instead of 3,000). An eager batch run over the whole subset is an optional extra, not a prerequisite. The mini evaluation is covered automatically, since its queries caption their own top results.

Before calling the vision model, check:

```text
cache/generated_image_descriptions.json
```

If the image was already processed, reuse the existing generated caption/tags.

Before generating embeddings, check:

```text
cache/image_embeddings.npy + cache/embeddings_meta.json
```

Reuse the cached embeddings only if **both** hold; otherwise recompute:

- `embeddings_meta.json` `"image_ids"` equals the `image_id` column of `metadata.csv` — same ids, same order;
- `embeddings_meta.json` `"model"` matches the current CLIP model name.

This check is mandatory: a stale cache loads silently and returns wrong images with plausible-looking scores — nothing crashes, so the bug is invisible at the demo.

This will make the notebook faster for demo.

---

# Display Requirements

In the notebook, after each search query, display:

- top-k images using matplotlib or PIL
- generated caption
- similarity score
- metadata
- explanation

Example layout:

```text
Rank 1 — img_014.jpg
Score: 0.31
Location: Almaty
Date: 2026-07-01
Caption: A dog runs across a grassy field.
Explanation: Matches dog + grass + running.
```

Then show the image.

---

# Optional Gradio UI

If there is time, add a simple Gradio interface.

Inputs:

```text
textbox: user query
slider: top_k
```

Outputs:

```text
gallery: matched images
json/text: explanations
```

Do not spend too much time on UI before the notebook search works.

---

# README Requirements

Create a short `README.md`.

It should include:

```md
# Image Extractor Agent

## What it does

This project searches a local image dataset using natural language. The original dataset is treated as images plus metadata only. Images are embedded locally with CLIP and searched semantically; captions and tags are generated with a vision model for explanations and reports. The project runs inside an Anaconda (conda) environment.

## Dataset

The dataset contains image files and `metadata.csv`.

Important: `captions.txt` is not used for indexing or retrieval.

## Pipeline

1. Load images and metadata
2. Generate captions and tags from images
3. Build searchable text
4. Generate embeddings
5. Search with natural language query
6. Explain matches
7. Save report

## How to run

```bash
conda create -n image-extractor python=3.11 -y
conda activate image-extractor
pip install -r requirements.txt
cp .env.example .env
# OPENAI_API_KEY is optional — only needed for caption generation
jupyter notebook image_extractor_agent_day5.ipynb
```

## Outputs

- `cache/generated_image_descriptions.json`
- `cache/image_embeddings.npy`
- `outputs/search_results.json`
- `outputs/search_report.md`

## Limitations

- Small dataset for demo
- Search quality depends on generated captions
- Metadata location/date may be synthetic
- The system retrieves likely matches but does not guarantee perfect visual understanding
```

---

# requirements.txt

Create:

```txt
torch  # install the CUDA build first: pip install torch --index-url https://download.pytorch.org/whl/cu128
open-clip-torch
kagglehub
python-dotenv
pandas
numpy
Pillow
matplotlib
scikit-learn
jupyter
```

Optional:

```txt
openai
gradio
```

Install into the conda environment (`conda activate image-extractor` first). For GPU acceleration install the CUDA build of PyTorch matching your driver.

---

# .env.example

Create:

```text
# Optional: only used for vision captions/tags and LLM explanations.
# CLIP retrieval works without any API key.
OPENAI_API_KEY=your_api_key_here
```

Do not commit real API keys.

---

# Implementation Order

Follow this order strictly:

## Step 1 — Prepare the dataset

- download Flickr8k with kagglehub, run scripts/prepare_flickr8k_subset.py --count 3000
- load metadata, display 5 sample images

## Step 2 — Generate CLIP image embeddings (no API key needed)

- does NOT depend on captions — runs with zero API calls
- develop on the first ~50 images, then embed all 3,000 (batched, torch.no_grad(), GPU)
- save image_embeddings.npy + embeddings_meta.json

## Step 3 — Implement search

- query → CLIP text encoder
- cosine similarity (one numpy matmul)
- return top-k

## Step 4 — Add metadata filters

- location, date
- filter BEFORE ranking (candidate mask over metadata.csv)

## Step 5 — Generate captions/tags (optional, needs OPENAI_API_KEY)

- lazy: caption only retrieved top-k images, accumulate in cache
- process 5 images first to validate the schema

## Step 6 — Build searchable text

- combine caption + tags + metadata (for explanations and the report only)

## Step 7 — Add explanation

- rule-based fallback from score + matched filters first
- LLM explanation optional

## Step 8 — Add agent wrapper

- parse query
- call tools
- return structured output

## Step 9 — Save report

- JSON output
- Markdown report

## Step 10 — Prepare demo

- 3–5 good queries
- show images
- show report
- explain architecture

---

# Acceptance Criteria

The project is complete when:

```text
[ ] The notebook runs end-to-end inside a conda (Anaconda) environment
[ ] It indexes the full 3,000-image Flickr8k subset
[ ] It generates captions/tags when OPENAI_API_KEY is set; without a key it stores the documented fallback records and retrieval, display, and the report still work end-to-end
[ ] It does not use captions.txt for indexing
[ ] It creates CLIP embeddings locally (image encoder for images, text encoder for queries)
[ ] It retrieves top-k images from a text query
[ ] It displays matching images
[ ] It shows metadata for each result
[ ] It explains why each result matched
[ ] It saves search_results.json
[ ] It saves search_report.md
[ ] It includes a short README
```

---

# What Not To Build Today

Do not spend time on:

```text
full backend
full frontend
authentication
database server
Docker
large-scale vector DB
video search
mobile app
complex deployment
```

Build the core image search first.

---

# 5-Minute Demo Script

Use this script during presentation:

```text
I built an Image Extractor Agent.

The goal is to search an image dataset using natural language, even when the dataset originally contains only image files and basic metadata.

I intentionally do not use the provided captions.txt file. Instead, the system embeds every image locally with CLIP, and a vision model generates captions and tags for explanations. The query is embedded with the CLIP text encoder and matched against the image vectors in a semantic search index.

When the user enters a query like “find images with a dog running outside,” the agent parses the query, searches the image index, applies optional metadata filters, returns the top matching images, explains why they match, and saves a markdown report.

Technically, the project combines vision-based image understanding, embeddings, semantic retrieval, and an agent-like tool workflow.
```

---

# Important Final Reminder

Start simple.

First make this work:

```text
query → top 5 images
```

Then add:

```text
explanation
metadata filters
report
agent wrapper
UI
```

A small working demo is better than a big unfinished system.
