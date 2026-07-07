import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from PIL import Image
from sentence_transformers import SentenceTransformer

load_dotenv()

ES_URL = os.environ.get("ES_URL", "")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
DATASET_DIR = Path("images")
INDEX_NAME = "image_embeddings"
MODEL_NAME = "jinaai/jina-clip-v2"


def parse_filename(path: Path) -> tuple[str, str]:
    match = re.match(r"([a-zA-Z]+)(\d+)", path.stem)
    if match:
        return match.group(1), match.group(2)
    return "unknown", path.stem


def create_index(es: Elasticsearch, dims: int) -> None:
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")

    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "category": {"type": "keyword"},
                "id": {"type": "keyword"},
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


def generate_actions(image_files: list[Path], embeddings):
    for path, emb in zip(image_files, embeddings):
        category, img_id = parse_filename(path)
        yield {
            "_index": INDEX_NAME,
            "_source": {
                "filename": path.name,
                "category": category,
                "id": img_id,
                "embedding": emb.tolist(),
            },
        }


def main():
    if not ES_URL or not ES_API_KEY:
        print("Error: ES_URL and ES_API_KEY environment variables must be set.")
        sys.exit(1)

    image_files = sorted(DATASET_DIR.glob("*.png"))
    if not image_files:
        print(f"No .png files found in '{DATASET_DIR}'")
        sys.exit(1)

    print(f"Found {len(image_files)} image files:")
    for f in image_files:
        print(f"  {f.name}")

    print(f"\nLoading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print("\nLoading images ...")
    images = [Image.open(f).convert("RGB") for f in image_files]

    print("Generating embeddings ...")
    embeddings = model.encode(images, show_progress_bar=True)
    dims = embeddings.shape[1]
    print(f"Embedding dimension: {dims}")

    print(f"\nConnecting to Elasticsearch at {ES_URL} ...")
    es = Elasticsearch(ES_URL, api_key=ES_API_KEY, verify_certs=False, ssl_show_warn=False)
    info = es.info()
    print(f"Connected to cluster: {info['cluster_name']}")

    create_index(es, dims)

    print("\nIndexing documents ...")
    success, errors = bulk(es, generate_actions(image_files, embeddings))
    print(f"Indexed {success} documents successfully.")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
