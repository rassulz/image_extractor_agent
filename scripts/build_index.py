"""Build (or refresh) the CLIP image-embedding index from metadata.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import build_index_if_needed  # noqa: E402

if __name__ == "__main__":
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    build_index_if_needed(batch_size=bs)
    print("Done.")
