"""Load metadata.csv and provide image path helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METADATA_CSV = ROOT / "data" / "metadata.csv"
IMAGES_DIR = ROOT / "data" / "images"


def load_metadata() -> pd.DataFrame:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"{METADATA_CSV} not found. Run scripts/prepare_flickr8k_subset.py first."
        )
    df = pd.read_csv(METADATA_CSV, dtype={"image_id": str})
    df = df.sort_values("image_id").reset_index(drop=True)
    return df


def image_abs_path(file_name: str) -> Path:
    return IMAGES_DIR / file_name
