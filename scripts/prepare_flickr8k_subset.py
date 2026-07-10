"""Prepare a deterministic Flickr8k subset + synthetic metadata.

Downloads Flickr8k via kagglehub, picks a deterministic subset (sorted
filenames + fixed seed), copies images into data/images/, and writes
data/metadata.csv with real width/height (PIL). Location is unknown (Flickr8k
has none); only a synthetic date is generated to demonstrate metadata filtering.

captions.txt is deliberately NOT copied and never used for indexing.

Usage:
    python scripts/prepare_flickr8k_subset.py --count 3000 --remove-raw
"""
from __future__ import annotations

import argparse
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "data" / "images"
METADATA_CSV = ROOT / "data" / "metadata.csv"

SEED = 42
# Flickr8k has no reliable capture location (it is an international dataset with
# unknown, unlabeled locations), so we do NOT invent one. Only a synthetic date
# is generated, purely to demonstrate metadata filtering.
START_DATE = date(2026, 7, 1)


def find_images_dir(raw_path: Path) -> Path:
    """Locate the folder that actually holds the .jpg files."""
    candidates = list(raw_path.rglob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No .jpg files found under {raw_path}")
    # Group by parent, use the folder with the most images.
    from collections import Counter

    parent_counts = Counter(p.parent for p in candidates)
    best_parent, _ = parent_counts.most_common(1)[0]
    return best_parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--remove-raw", action="store_true")
    args = parser.parse_args()

    import kagglehub

    print("Downloading Flickr8k via kagglehub ...")
    raw_path = Path(kagglehub.dataset_download("adityajn105/flickr8k"))
    print("Dataset at:", raw_path)

    src_dir = find_images_dir(raw_path)
    all_images = sorted(p.name for p in src_dir.glob("*.jpg"))
    print(f"Found {len(all_images)} images in {src_dir}")

    rng = random.Random(SEED)
    if args.count < len(all_images):
        chosen = sorted(rng.sample(all_images, args.count))
    else:
        chosen = all_images
    print(f"Selected {len(chosen)} images (seed={SEED})")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname in chosen:
        src = src_dir / fname
        dst = IMAGES_DIR / fname
        try:
            with Image.open(src) as im:
                w, h = im.size
        except Exception as e:  # noqa: BLE001
            print(f"  skip corrupt image {fname}: {e}")
            continue
        if not dst.exists():
            shutil.copy2(src, dst)

        image_id = Path(fname).stem
        d = START_DATE + timedelta(days=rng.randrange(0, 60))
        rows.append(
            {
                "image_id": image_id,
                "file_name": fname,
                "image_path": f"data/images/{fname}",
                "location": "unknown",
                "date": d.isoformat(),
                "width": w,
                "height": h,
                "source": "flickr8k",
            }
        )

    rows.sort(key=lambda r: r["image_id"])

    import csv

    with METADATA_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "image_path",
                "location",
                "date",
                "width",
                "height",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {METADATA_CSV}")
    print("NOTE: location is unknown (Flickr8k has none); only 'date' is synthetic (seeded RNG).")

    if args.remove_raw:
        try:
            shutil.rmtree(raw_path)
            print("Removed raw download.")
        except Exception as e:  # noqa: BLE001
            print(f"Could not remove raw download: {e}")


if __name__ == "__main__":
    main()
