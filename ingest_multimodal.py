import os
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
DATASET_DIR = Path("multimodal")
INDEX_NAME = "multimodal-embeddings"
MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small"
NUM_FRAMES = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4"}
TEXT_EXTS = {".txt"}


def extract_video_frames(video_path: Path) -> list[str]:
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
                "file_path": {"type": "keyword"},
                "media_type": {"type": "keyword"},
                "content": {"type": "text"},
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


def collect_files() -> tuple[list[Path], list[Path], list[Path]]:
    images, videos, texts = [], [], []
    for path in sorted(DATASET_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(path)
        elif ext in VIDEO_EXTS:
            videos.append(path)
        elif ext in TEXT_EXTS:
            texts.append(path)
    return images, videos, texts


def generate_actions(records: list[dict]):
    for rec in records:
        yield {"_index": INDEX_NAME, "_source": rec}


def main():
    if not ES_URL or not ES_API_KEY:
        print("Error: ES_URL and ES_API_KEY environment variables must be set.")
        sys.exit(1)

    images, videos, texts = collect_files()
    print(f"Found {len(images)} images, {len(videos)} videos, {len(texts)} texts")

    print(f"\nLoading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    records = []

    # --- Images ---
    if images:
        print(f"\nEmbedding {len(images)} images ...")
        img_paths = [str(p) for p in images]
        img_embeddings = model.encode(img_paths, task='retrieval', show_progress_bar=True)
        for path, emb in zip(images, img_embeddings):
            records.append({
                "filename": path.name,
                "file_path": str(path),
                "media_type": "image",
                "content": None,
                "embedding": emb.tolist(),
            })

    # --- Videos ---
    if videos:
        print(f"\nEmbedding {len(videos)} videos ({NUM_FRAMES} frames each) ...")
        for path in videos:
            print(f"  Processing: {path.name}")
            frame_paths = extract_video_frames(path)
            if not frame_paths:
                print(f"    Warning: no frames extracted — skipping {path.name}")
                continue
            frame_embs = model.encode(frame_paths, task='retrieval', show_progress_bar=False)
            avg_emb = frame_embs.mean(axis=0)
            for fp in frame_paths:
                Path(fp).unlink(missing_ok=True)
            records.append({
                "filename": path.name,
                "file_path": str(path),
                "media_type": "video",
                "content": None,
                "embedding": avg_emb.tolist(),
            })

    # --- Texts ---
    if texts:
        print(f"\nEmbedding {len(texts)} text files ...")
        text_contents = [p.read_text(encoding="utf-8", errors="ignore").strip() for p in texts]
        text_embeddings = model.encode(text_contents, task='retrieval', show_progress_bar=True)
        for path, content, emb in zip(texts, text_contents, text_embeddings):
            records.append({
                "filename": path.name,
                "file_path": str(path),
                "media_type": "text",
                "content": content,
                "embedding": emb.tolist(),
            })

    if not records:
        print("No files found to index.")
        sys.exit(1)

    dims = len(records[0]["embedding"])
    print(f"\nEmbedding dimension: {dims}")

    print(f"\nConnecting to Elasticsearch at {ES_URL} ...")
    es = Elasticsearch(ES_URL, api_key=ES_API_KEY, verify_certs=False, ssl_show_warn=False)
    info = es.info()
    print(f"Connected to cluster: {info['cluster_name']}")

    create_index(es, dims)

    print("\nIndexing documents ...")
    success, errors = bulk(es, generate_actions(records))
    print(f"Indexed {success} documents successfully.")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
