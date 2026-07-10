#!/usr/bin/env python3
"""Keep a reproducible, caption-free Flickr8k subset for the Day 5 demo.

Run after Kaggle download:
    python scripts/prepare_flickr8k_subset.py --remove-raw
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import IMAGE_SUFFIXES, create_synthetic_metadata, image_size  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local 3,000-image Flickr8k subset without captions.txt.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/flickr8k")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data/images")
    parser.add_argument("--metadata-path", type=Path, default=PROJECT_ROOT / "data/metadata.csv")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace any existing image files in --images-dir.")
    parser.add_argument("--remove-raw", action="store_true", help="Delete the full Kaggle download after a successful subset build.")
    return parser.parse_args()


def image_candidates(raw_dir: Path) -> list[Path]:
    return sorted(
        (path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().lower(),
    )


def remove_caption_files(directory: Path) -> list[Path]:
    removed: list[Path] = []
    for candidate in directory.rglob("captions.txt"):
        if candidate.is_file():
            candidate.unlink()
            removed.append(candidate)
    return removed


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    images_dir = args.images_dir.resolve()
    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw Kaggle dataset not found: {raw_dir}")

    candidates = image_candidates(raw_dir)
    random.Random(args.seed).shuffle(candidates)
    valid: list[Path] = []
    for candidate in candidates:
        if image_size(candidate) is not None:
            valid.append(candidate)
        if len(valid) == args.count:
            break
    if len(valid) < args.count:
        raise ValueError(f"Only {len(valid)} readable images were found; expected {args.count}.")

    images_dir.mkdir(parents=True, exist_ok=True)
    existing_images = [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if existing_images and not args.overwrite:
        raise FileExistsError(
            f"{images_dir} already contains {len(existing_images)} images. Use --overwrite to replace them."
        )
    if args.overwrite:
        for existing in existing_images:
            existing.unlink()

    for source in valid:
        destination = images_dir / source.name
        if destination.exists():
            raise FileExistsError(f"Duplicate image name encountered: {destination.name}")
        shutil.copy2(source, destination)

    metadata = create_synthetic_metadata(images_dir, args.metadata_path, source="flickr8k", seed=args.seed)
    removed_captions = remove_caption_files(raw_dir)
    manifest = {
        "dataset": "adityajn105/flickr8k",
        "selection_method": "deterministic random sample of readable image files",
        "seed": args.seed,
        "image_count": len(metadata),
        "captions_used_for_indexing": False,
        "captions_txt_removed": [path.relative_to(PROJECT_ROOT).as_posix() for path in removed_captions],
    }
    (PROJECT_ROOT / "data/subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.remove_raw:
        shutil.rmtree(raw_dir)

    print(f"Created {len(metadata)} images in {images_dir}")
    print(f"Created metadata: {args.metadata_path}")
    print("captions.txt is not in the working subset and is not used for indexing.")


if __name__ == "__main__":
    main()
