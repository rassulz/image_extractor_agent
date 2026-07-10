# Task for VS Code LLM: Build Image Extractor Agent for YDL 2026 Day 5 Capstone

## 0. Context

We are building a **YDL 2026 Day 5 Capstone project**, not a GovTech product.

The goal of Day 5 is to build a small but working AI system that can be demoed in 5 minutes. The project should run end-to-end in a notebook, use at least one agent/tool/RAG-style pattern, produce a visible artifact, and be easy to explain.

## 1. Project Name

**Image Extractor Agent**

## 2. Project Goal

Build an AI agent that receives a natural-language description of an image and searches a local image dataset to find the most relevant images.

The dataset initially contains only:

- image file
- location
- date
- image size / resolution
- optional source field

The system must enrich the images by extracting visual captions/tags, build a searchable index, and allow users to search the dataset using normal language.

Example user queries:

```text
Find images with cars near buildings.
Find photos taken in Almaty with trees and pedestrians.
Find images that look like an urban street during the day.
Find pictures with people sitting indoors.
Find images from July that contain roads or traffic.
```

The system should return:

- top matching images
- metadata for each image
- similarity score
- generated caption/tags
- explanation of why each image matches the query
- saved markdown or JSON report

## 3. Important Scope Rule

Do **not** build a full production app.
Do **not** build a backend/frontend unless the core notebook is already complete.
Do **not** over-engineer.

The priority is:

1. Working notebook
2. Working image search
3. Clear agent/tool structure
4. Search result explanations
5. Saved output artifact
6. Optional simple UI only if time remains

## 4. Expected Final Demo

The notebook should demonstrate this flow:

```text
1. Load image dataset and metadata
2. Generate or load image captions/tags
3. Build searchable image index
4. User enters a text query
5. Agent searches the image index
6. Agent returns top-k images with metadata and explanations
7. System saves a report to outputs/search_report.md or outputs/search_results.json
```

The demo should be possible in 5 minutes.

## 5. Recommended Tech Stack

Use Python notebook.

Required libraries:

```text
openai
pandas
numpy
scikit-learn
Pillow
matplotlib
python-dotenv
```

Optional libraries:

```text
gradio
faiss-cpu
chromadb
```

Use the simplest reliable implementation first.

For vector search, start with:

```python
sklearn.metrics.pairwise.cosine_similarity
```

Do not use FAISS/Chroma unless the simple local version already works.

## 6. Project Structure

Create or use this structure:

```text
image-extractor-agent/
│
├── image_extractor_day5.ipynb
├── README.md
├── .env.example
│
├── data/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── img_003.jpg
│   │
│   └── metadata.csv
│
├── outputs/
│   ├── image_index.json
│   ├── search_results.json
│   └── search_report.md
│
└── src/                       # optional, only if notebook gets too long
    ├── dataset.py
    ├── vision.py
    ├── embeddings.py
    ├── search.py
    ├── agent_tools.py
    └── reporting.py
```

For Day 5, the notebook alone is enough. Use `src/` only if it helps keep things clean.

## 7. Dataset Requirements

### 7.1 Minimum Dataset Size

Use at least:

```text
20–50 images
```

Better if available:

```text
100–300 images
```

Do not spend too much time collecting data. A small, clean dataset is better than a large messy one.

### 7.2 Image Types

Prefer a focused image theme. Recommended option:

**Urban/general visual dataset**

Include images such as:

```text
streets
cars
buildings
parks
trees
people
roads
traffic lights
indoor rooms
food/table scenes
classrooms/offices
```

The dataset can also be personal/local images as long as they are safe to use and demo.

### 7.3 metadata.csv Format

Create `data/metadata.csv` with this minimum schema:

```csv
image_id,file_name,location,date,width,height,source
img_001,img_001.jpg,Almaty,2026-07-01,1024,768,local
img_002,img_002.jpg,Astana,2026-06-21,1280,720,local
img_003,img_003.jpg,Atyrau,2026-05-14,800,600,local
```

Required columns:

- `image_id` — unique ID
- `file_name` — image filename inside `data/images/`
- `location` — city/place label, can be approximate
- `date` — ISO date format if possible, e.g. `2026-07-01`
- `width` — image width in pixels
- `height` — image height in pixels
- `source` — `local`, `unsplash`, `kaggle`, `generated`, etc.

If width/height are missing, calculate them with PIL.

### 7.4 Enriched Index Schema

After processing, create `outputs/image_index.json` with this structure:

```json
[
  {
    "image_id": "img_001",
    "file_name": "img_001.jpg",
    "image_path": "data/images/img_001.jpg",
    "location": "Almaty",
    "date": "2026-07-01",
    "width": 1024,
    "height": 768,
    "caption": "A city street with several cars parked near a modern building.",
    "objects": ["cars", "street", "building", "trees"],
    "scene": "urban street",
    "search_text": "A city street with several cars parked near a modern building. Objects: cars, street, building, trees. Scene: urban street. Location: Almaty. Date: 2026-07-01.",
    "embedding": [0.012, -0.034, 0.056]
  }
]
```

If embedding vectors make the JSON too large, store embeddings in memory during the notebook run and save only metadata/captions. But for a small dataset, JSON is acceptable.

## 8. AI Components

The system should have these AI/data components:

### 8.1 Vision Caption Extraction

For each image, generate:

- caption
- objects
- scene
- visual features

Example:

```json
{
  "caption": "A wide road with several cars and tall buildings in the background.",
  "objects": ["road", "cars", "buildings", "sky"],
  "scene": "urban road",
  "visual_features": ["outdoor", "daylight", "traffic"]
}
```

Preferred implementation:

- Use OpenAI Vision API if API key is available.

Fallback implementation:

- Use manually prepared captions in a CSV/JSON file.
- Or create simple placeholder captions for demo images.

Important: the project must still run even if vision extraction is skipped by loading cached captions from `outputs/image_index.json`.

### 8.2 Embedding Generation

Create embeddings for each image using its searchable text:

```text
caption + objects + scene + location + date
```

Also create embedding for the user query.

Use OpenAI embeddings if available.

Fallback:

- Use `TfidfVectorizer` from scikit-learn for local semantic-ish search.

Recommended approach:

1. Implement OpenAI embeddings.
2. Add fallback to TF-IDF if no API key is available.

### 8.3 Vector Search

Use cosine similarity to compare query embedding against image embeddings.

Return top-k results.

Default:

```python
top_k = 5
```

### 8.4 Metadata Filtering

Support simple filters if the query mentions them:

- location
- date/year/month
- image size if easy

For MVP, the filter can be simple and explicit.

Example:

```python
search_images(
    query="cars near buildings",
    location="Almaty",
    top_k=5
)
```

Do not spend too much time on perfect natural-language date parsing.

### 8.5 Explanation Generation

For every returned image, provide an explanation:

```text
This image matches because the generated caption mentions cars and buildings, and the scene is classified as an urban street. The metadata also matches the requested location: Almaty.
```

Can be generated by:

- LLM
- or a deterministic template using query, caption, tags, and metadata

For reliability, start with template explanation. Then optionally add LLM explanation.

## 9. Agent Design

The agent should be a lightweight orchestration layer around tools.

### 9.1 Agent Role

The agent receives a user query and decides how to use tools to retrieve and explain image matches.

Agent instruction:

```text
You are an Image Extractor Agent. Your job is to help users find images from a local dataset using natural language. You must use available tools to parse the query, search the image index, apply metadata filters if needed, explain the matches, and save a report. Do not invent images. Only return images that exist in the dataset.
```

### 9.2 Tools to Implement

Implement these Python functions as tools:

```python
def load_image_index(index_path: str = "outputs/image_index.json"):
    """Load enriched image index from disk."""


def parse_user_query(query: str) -> dict:
    """Extract visual intent and optional filters from user query."""


def search_images(query: str, top_k: int = 5, location: str | None = None, date: str | None = None) -> list[dict]:
    """Search image index and return top-k matching images."""


def explain_results(query: str, results: list[dict]) -> list[dict]:
    """Add explanation for each image match."""


def save_report(query: str, results: list[dict], output_path: str = "outputs/search_report.md") -> str:
    """Save markdown report and return output path."""
```

Optional:

```python
def display_results(results: list[dict]):
    """Display images with metadata inside notebook."""
```

### 9.3 Agent Flow

Expected flow:

```text
User query
→ parse_user_query
→ search_images
→ explain_results
→ save_report
→ final answer
```

The final answer should include:

- interpreted query
- number of results found
- result list
- report path

## 10. Notebook Sections

Create `image_extractor_day5.ipynb` with these sections:

### Section 1 — Setup

- imports
- load `.env`
- configure OpenAI client if API key exists
- create required folders

### Section 2 — Load Dataset

- load metadata.csv
- validate required columns
- calculate width/height if missing
- show dataframe preview

### Section 3 — Show Sample Images

- display 3–5 sample images
- show metadata

### Section 4 — Caption Extraction

- generate captions/tags for images
- cache results
- if cached index exists, load it to avoid repeated API calls

### Section 5 — Build Search Text

Create `search_text` field:

```text
Caption: ... Objects: ... Scene: ... Location: ... Date: ...
```

### Section 6 — Build Embeddings / TF-IDF Index

- OpenAI embeddings if API key exists
- fallback to TF-IDF

### Section 7 — Search Function

Implement:

```python
search_images("cars near buildings", top_k=5)
```

### Section 8 — Explanation Function

Add readable explanations.

### Section 9 — Agent Wrapper

Implement simple agent orchestration.

May be either:

- OpenAI Agents SDK if available
- or a Python function that behaves like an agent using tools

For Day 5, a clean Python orchestration function is acceptable if the full Agents SDK becomes slow to integrate.

Example:

```python
def run_image_extractor_agent(user_query: str, top_k: int = 5):
    parsed = parse_user_query(user_query)
    results = search_images(
        query=parsed["visual_query"],
        top_k=top_k,
        location=parsed.get("location"),
        date=parsed.get("date")
    )
    explained = explain_results(user_query, results)
    report_path = save_report(user_query, explained)
    return {
        "query": user_query,
        "parsed": parsed,
        "results": explained,
        "report_path": report_path
    }
```

### Section 10 — Demo Queries

Run at least 3 demo queries:

```python
run_image_extractor_agent("Find images with cars near buildings", top_k=5)
run_image_extractor_agent("Find photos with trees and people walking", top_k=5)
run_image_extractor_agent("Find images from Almaty that look like urban streets", top_k=5)
```

### Section 11 — Save Final Artifact

Save:

```text
outputs/search_report.md
outputs/search_results.json
outputs/image_index.json
```

### Section 12 — Mini README in Notebook

Add a markdown cell explaining:

- what the project does
- architecture
- tools used
- limitations
- how to run

## 11. Output Format

### 11.1 Search Result JSON

Each result should look like:

```json
{
  "rank": 1,
  "image_id": "img_014",
  "file_name": "img_014.jpg",
  "image_path": "data/images/img_014.jpg",
  "location": "Almaty",
  "date": "2026-07-01",
  "width": 1024,
  "height": 768,
  "caption": "Cars parked near a modern building on a city street.",
  "objects": ["cars", "building", "street"],
  "scene": "urban street",
  "similarity_score": 0.91,
  "explanation": "This image matches because it contains cars, a building, and an urban street scene."
}
```

### 11.2 Markdown Report Format

Generate `outputs/search_report.md`:

```markdown
# Image Search Report

## Query

Find images with cars near buildings.

## Top Matches

### 1. img_014.jpg

- Location: Almaty
- Date: 2026-07-01
- Size: 1024x768
- Similarity score: 0.91
- Caption: Cars parked near a modern building on a city street.
- Explanation: This image matches because it contains cars, a building, and an urban street scene.

Image path: `data/images/img_014.jpg`
```

## 12. README Requirements

Create `README.md` with:

```markdown
# Image Extractor Agent

A Day 5 YDL 2026 capstone project that searches a local image dataset using natural language.

## Features

- Loads local images and metadata
- Extracts image captions and visual tags
- Builds a searchable index
- Searches images by natural-language description
- Returns top matching images with metadata
- Explains why each image matches
- Saves a markdown report

## How to Run

1. Put images into `data/images/`
2. Create `data/metadata.csv`
3. Add OpenAI API key to `.env` if using OpenAI models
4. Open `image_extractor_day5.ipynb`
5. Run all cells

## Dataset Format

See `data/metadata.csv`.

## Demo Queries

- Find images with cars near buildings
- Find photos with trees and people walking
- Find images from Almaty that look like urban streets

## Limitations

- Small local dataset
- Caption quality depends on vision model
- Search depends on captions and embeddings
- Metadata filtering is simple
```

## 13. .env.example

Create `.env.example`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The notebook should not crash if the key is missing. It should fall back to cached captions or TF-IDF search.

## 14. Acceptance Criteria

The task is complete when:

```text
[ ] Notebook runs end-to-end
[ ] Dataset metadata loads correctly
[ ] At least 20 images are searchable
[ ] Captions/tags exist for images
[ ] Search function returns top-k image matches
[ ] Results include metadata
[ ] Results include explanations
[ ] Markdown report is saved
[ ] JSON results are saved
[ ] README exists
[ ] Demo can be presented in 5 minutes
```

## 15. Important Implementation Advice

Build in this order:

```text
1. Load metadata and images
2. Display sample images
3. Create/load captions
4. Build search_text
5. Implement embeddings or TF-IDF
6. Implement search_images
7. Add explanations
8. Save report
9. Wrap with agent-like function
10. Optional: add OpenAI Agents SDK / trace / UI
```

Do not start with complex agent logic.

First make the search work.
Then wrap it as an agent.

## 16. Optional Enhancements

Only add these after MVP works:

```text
[ ] Gradio UI
[ ] OpenAI Agents SDK trace
[ ] Better query parser with LLM
[ ] Date/location extraction from natural language
[ ] CLIP-based image-text similarity
[ ] Evaluation with 3–5 test queries
[ ] Thumbnail grid display
```

## 17. Suggested Demo Script

Use this short explanation during presentation:

```text
I built an Image Extractor Agent. The problem is that image datasets often contain only images and basic metadata, so they are hard to search using natural language.

My system first enriches every image by generating a caption, visual tags, and searchable text. Then it builds embeddings and allows a user to search the dataset with a normal sentence.

The agent uses tools for query parsing, image retrieval, result explanation, and report generation. The output is a ranked list of matching images with metadata, similarity scores, explanations, and a saved markdown report.
```

## 18. Final Instruction to Coding LLM

Please implement the MVP first. The final deliverable should be a working `image_extractor_day5.ipynb` notebook plus `README.md`, `.env.example`, and output files.

Focus on correctness, simplicity, and demoability. Avoid production-level complexity.
