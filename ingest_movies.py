import os
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

load_dotenv()

ES_URL = os.environ.get("ES_URL", "")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
DATASET_DIR = Path("movies")
INDEX_NAME = "movie_embeddings"
MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small-retrieval"
NUM_FRAMES = 8  # evenly-spaced frames sampled per video

FRANCHISE_KEYWORDS = {
    "Star Wars": ["star wars", "jakku", "jedi", "sith", "lightsaber"],
    "The Matrix": ["matrix"],
}


def detect_franchise(filename: str) -> str:
    lower = filename.lower()
    for franchise, keywords in FRANCHISE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return franchise
    return "Unknown"


def clean_title(path: Path) -> str:
    name = path.stem
    # Strip trailing " | ..." or " - ..." attributions
    name = re.sub(r"\s*[|\-–]\s*.*$", "", name).strip()
    return name


def extract_frames(video_path: Path) -> list[str]:
    """Sample NUM_FRAMES evenly across the video, save as temp PNGs, return their paths."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

    tmp_paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, frame)
        tmp_paths.append(tmp.name)

    cap.release()
    return tmp_paths


def create_index(es: Elasticsearch, dims: int) -> None:
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")

    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "franchise": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index '{INDEX_NAME}' with {dims}-dim vectors")


def generate_actions(video_files: list[Path], embeddings: list):
    for path, emb in zip(video_files, embeddings):
        yield {
            "_index": INDEX_NAME,
            "_source": {
                "filename": path.name,
                "title": clean_title(path),
                "franchise": detect_franchise(path.name),
                "embedding": emb.tolist(),
            },
        }


def main():
    if not ES_URL or not ES_API_KEY:
        print("Error: ES_URL and ES_API_KEY environment variables must be set.")
        sys.exit(1)

    video_files = sorted(DATASET_DIR.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in '{DATASET_DIR}'")
        sys.exit(1)

    print(f"Found {len(video_files)} video files:")
    for f in video_files:
        print(f"  {f.name}")

    print(f"\nLoading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print(f"\nExtracting {NUM_FRAMES} frames per video and generating embeddings ...")
    all_embeddings = []

    for video_path in video_files:
        print(f"  Processing: {video_path.name}")
        frame_paths = extract_frames(video_path)

        if not frame_paths:
            print(f"    Warning: no frames extracted — skipping {video_path.name}")
            continue

        frame_embeddings = model.encode(frame_paths, show_progress_bar=False)
        all_embeddings.append(frame_embeddings.mean(axis=0))

        for p in frame_paths:
            Path(p).unlink(missing_ok=True)

    if not all_embeddings:
        print("No embeddings generated.")
        sys.exit(1)

    dims = all_embeddings[0].shape[0]
    print(f"Embedding dimension: {dims}")

    print(f"\nConnecting to Elasticsearch at {ES_URL} ...")
    es = Elasticsearch(ES_URL, api_key=ES_API_KEY, verify_certs=False, ssl_show_warn=False)
    info = es.info()
    print(f"Connected to cluster: {info['cluster_name']}")

    create_index(es, dims)

    print("\nIndexing documents ...")
    success, errors = bulk(es, generate_actions(video_files, all_embeddings))
    print(f"Indexed {success} documents successfully.")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
