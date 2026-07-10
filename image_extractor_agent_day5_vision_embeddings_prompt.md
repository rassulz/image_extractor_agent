# Image Extractor Agent — Day 5 Capstone Implementation Prompt

## Context

We are building a **Day 5 Capstone project** for an AI Engineering course.

The project is an **Image Extractor Agent**.

The goal is to search a local image dataset using natural language, even when the original dataset contains only:

- image files
- basic metadata

The key pipeline is:

```text
Images + metadata
→ Vision caption/tag extraction
→ Searchable text generation
→ Embeddings
→ Semantic image retrieval
→ Agent explanation
→ Saved report
```

The project must be implemented as a working notebook/demo, not as a production app.

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
Images folder
     │
     ▼
Metadata loader
     │
     ▼
Vision extractor
     │
     ├── generated caption
     ├── object tags
     ├── scene tags
     ├── activity tags
     └── visual attributes
     │
     ▼
Searchable text builder
     │
     ▼
Embedding generator
     │
     ▼
Local vector index
```

---

# Recommended Tech Stack

Use Python notebook first.

Required:

```text
python
pandas
numpy
Pillow
matplotlib
scikit-learn
openai
python-dotenv
```

Optional:

```text
gradio
faiss-cpu
chromadb
```

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
├── data/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   │
│   └── metadata.csv
│
├── cache/
│   ├── generated_image_descriptions.json
│   └── image_embeddings.npy
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

The minimum dataset should contain:

```text
20–50 images
```

Better:

```text
100–300 images
```

Do not use a huge dataset for the Day 5 demo.

The images can come from:

- Flickr8k subset
- personal image folder
- Unsplash subset
- any local folder of mixed images

For the first working demo, use a small subset.

---

## Required Metadata File

Create `data/metadata.csv`.

Required columns:

```csv
image_id,file_name,image_path,location,date,width,height,source
```

Example:

```csv
img_001,img_001.jpg,data/images/img_001.jpg,Almaty,2026-07-01,1024,768,flickr8k
img_002,img_002.jpg,data/images/img_002.jpg,Astana,2026-07-02,800,600,flickr8k
img_003,img_003.jpg,data/images/img_003.jpg,Atyrau,2026-07-03,1280,720,flickr8k
```

If real location/date are not available, generate realistic synthetic metadata.

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

This `searchable_text` is what should be embedded.

Do not embed raw metadata only.

Do not embed `captions.txt`.

---

# Embedding and Search

## Embedding Step

For each image:

```text
searchable_text → embedding vector
```

Save embeddings to:

```text
cache/image_embeddings.npy
```

Save enriched metadata to:

```text
cache/generated_image_descriptions.json
```

## Query Step

For user query:

```text
user query → query embedding
```

Then calculate cosine similarity:

```text
similarity(query_embedding, image_embedding)
```

Return top-k results.

Recommended default:

```text
top_k = 5
```

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

---

# Agent Requirements

The agent should behave like an orchestrator.

It should not manually perform all logic inside one giant function.

Create small tools/functions:

```python
def parse_query_tool(query: str) -> dict:
    ...

def search_images_tool(visual_query: str, top_k: int = 5) -> list:
    ...

def metadata_filter_tool(results: list, filters: dict) -> list:
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
3. Search image index using semantic embeddings
4. Apply metadata filters
5. Rank results
6. Explain each result
7. Save report
8. Return final structured answer
```

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
      "similarity_score": 0.91,
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
- Similarity Score: 0.91
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

Required sections:

```text
1. Project overview
2. Setup and imports
3. Load environment variables
4. Load image dataset
5. Create or load metadata.csv
6. Display sample images
7. Generate image captions and tags with vision model
8. Build searchable text
9. Generate embeddings
10. Build local vector index
11. Implement semantic search
12. Implement metadata filters
13. Implement agent tools
14. Run agent demo queries
15. Save JSON and Markdown reports
16. Mini evaluation
17. Demo script / explanation
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
dog running on grass,img_014,0.91,good,Dog and grass clearly visible
people near water,img_022,0.88,good,People standing near lake
bicycle on street,img_031,0.84,partial,Bicycle present but not clearly on street
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

Vision extraction and embeddings can cost time and API credits.

Therefore, implement caching.

Before calling the vision model, check:

```text
cache/generated_image_descriptions.json
```

If the image was already processed, reuse the existing generated caption/tags.

Before generating embeddings, check:

```text
cache/image_embeddings.npy
```

If embeddings already exist and match the current image index, reuse them.

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
Score: 0.91
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

This project searches a local image dataset using natural language. The original dataset is treated as images plus metadata only. Captions are generated automatically using a vision model, embedded, and searched semantically.

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
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY to .env
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
openai
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
gradio
```

---

# .env.example

Create:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit real API keys.

---

# Implementation Order

Follow this order strictly:

## Step 1 — Get dataset loading working

- load metadata
- load images
- display 5 sample images

## Step 2 — Generate captions/tags

- process 5 images first
- save to cache
- then scale to 20–50 images

## Step 3 — Build searchable text

- combine caption + tags + metadata

## Step 4 — Generate embeddings

- embed searchable text
- save vectors

## Step 5 — Implement search

- embed query
- cosine similarity
- return top-k

## Step 6 — Add metadata filters

- location
- date

## Step 7 — Add explanation

- simple explanation first
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
[ ] The notebook runs end-to-end
[ ] It loads at least 20 images
[ ] It does not use captions.txt for indexing
[ ] It generates captions/tags from images
[ ] It creates embeddings
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

I intentionally do not use the provided captions.txt file. Instead, the system analyzes each image with a vision model, generates captions and tags, builds embeddings, and creates a semantic search index.

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
