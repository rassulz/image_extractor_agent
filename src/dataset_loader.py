"""Dataset and metadata helpers for the Image Extractor Agent."""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_METADATA_COLUMNS = [
    "image_id",
    "file_name",
    "image_path",
    "location",
    "date",
    "width",
    "height",
    "source",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SYNTHETIC_LOCATIONS = ["Almaty", "Astana", "Atyrau", "Shymkent", "Aktau", "Karaganda"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_project_directories(root: Path | None = None) -> None:
    root = root or project_root()
    for relative_path in ("data/images", "data/raw", "cache", "outputs"):
        (root / relative_path).mkdir(parents=True, exist_ok=True)


def discover_image_files(images_dir: str | Path) -> list[Path]:
    """Return image files in a deterministic order."""
    directory = Path(images_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Images directory does not exist: {directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def image_size(image_path: str | Path) -> tuple[int, int] | None:
    """Read image dimensions, returning None for unreadable files."""
    try:
        with Image.open(image_path) as image:
            return image.size
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None


def create_synthetic_metadata(
    images_dir: str | Path = "data/images",
    metadata_path: str | Path = "data/metadata.csv",
    *,
    source: str = "flickr8k",
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic *synthetic* place/date metadata for local images.

    Flickr8k does not provide reliable location or capture date. The generated
    values are explicitly marked in ``metadata_note`` so they are never confused
    with real EXIF information.
    """
    image_paths = discover_image_files(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}. Add images before creating metadata.")

    rng = random.Random(seed)
    start_date = date(2026, 7, 1)
    rows: list[dict[str, object]] = []
    for number, path in enumerate(image_paths, start=1):
        dimensions = image_size(path)
        if dimensions is None:
            continue
        width, height = dimensions
        rows.append(
            {
                "image_id": f"img_{number:04d}",
                "file_name": path.name,
                "image_path": path.as_posix(),
                "location": rng.choice(SYNTHETIC_LOCATIONS),
                "date": (start_date + timedelta(days=(number - 1) % 31)).isoformat(),
                "width": width,
                "height": height,
                "source": source,
                "metadata_note": "location and date are synthetic for the Day 5 demo",
            }
        )

    metadata = pd.DataFrame(rows)
    output = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return metadata


def validate_metadata(metadata: pd.DataFrame, *, check_files: bool = True) -> pd.DataFrame:
    """Validate the public dataset schema and normalize basic types."""
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {', '.join(missing)}")
    if metadata.empty:
        raise ValueError("metadata.csv is empty. Create it from data/images first.")
    if metadata["image_id"].isna().any() or metadata["image_id"].duplicated().any():
        raise ValueError("image_id values must be present and unique.")
    if metadata["file_name"].isna().any() or metadata["file_name"].duplicated().any():
        raise ValueError("file_name values must be present and unique.")

    normalized = metadata.copy()
    for column in ("image_id", "file_name", "image_path", "location", "date", "source"):
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date.astype(str)
    normalized["width"] = pd.to_numeric(normalized["width"], errors="raise").astype(int)
    normalized["height"] = pd.to_numeric(normalized["height"], errors="raise").astype(int)
    if (normalized[["width", "height"]] <= 0).any().any():
        raise ValueError("width and height must be positive numbers.")

    if check_files:
        missing_files = [path for path in normalized["image_path"] if not Path(path).is_file()]
        if missing_files:
            sample = ", ".join(missing_files[:3])
            raise FileNotFoundError(f"{len(missing_files)} image paths are missing (for example: {sample}).")
    return normalized


def load_metadata(metadata_path: str | Path = "data/metadata.csv", *, check_files: bool = True) -> pd.DataFrame:
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Run create_synthetic_metadata after adding images."
        )
    return validate_metadata(pd.read_csv(path), check_files=check_files)


def build_searchable_text(record: dict[str, object]) -> str:
    """Combine generated visual description and metadata into embedding text."""
    def values(name: str) -> str:
        value = record.get(name, [])
        if isinstance(value, str):
            return value
        return ", ".join(str(item) for item in value) or "unknown"

    return "\n".join(
        [
            f"Image ID: {record['image_id']}.",
            f"Caption: {record.get('generated_caption', 'No caption generated.')}",
            f"Objects: {values('objects')}.",
            f"Scene: {record.get('scene', 'unknown')}.",
            f"Activities: {values('activities')}.",
            f"Visual attributes: {values('visual_attributes')}.",
            f"Location: {record.get('location', 'unknown')}.",
            f"Date: {record.get('date', 'unknown')}.",
            f"Size: {record.get('width', 'unknown')}x{record.get('height', 'unknown')}.",
        ]
    )


def records_from_metadata(metadata: pd.DataFrame) -> list[dict[str, object]]:
    """Convert normalized metadata rows to JSON-friendly dictionaries."""
    return metadata.where(pd.notna(metadata), None).to_dict(orient="records")
