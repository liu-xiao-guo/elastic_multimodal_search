import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

load_dotenv()

ES_URL = os.environ.get("ES_URL", "")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
DATASET_DIR = Path("animals")
MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small-retrieval"

MATRYOSHKA_DIMS = [128, 256, 512, 1024]


def index_name(dims: int) -> str:
    return f"matryoshka_{dims}_embeddings"


def parse_filename(path: Path) -> tuple[str, str]:
    match = re.match(r"([a-zA-Z]+)(\d+)", path.stem)
    if match:
        return match.group(1), match.group(2)
    return "unknown", path.stem


def create_index(es: Elasticsearch, name: str, dims: int) -> None:
    if es.indices.exists(index=name):
        es.indices.delete(index=name)
        print(f"Deleted existing index '{name}'")

    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "animal": {"type": "keyword"},
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
    es.indices.create(index=name, body=mapping)
    print(f"Created index '{name}' with {dims}-dim vectors")


def generate_actions(name: str, image_files: list[Path], embeddings):
    for path, emb in zip(image_files, embeddings):
        animal, animal_id = parse_filename(path)
        yield {
            "_index": name,
            "_source": {
                "filename": path.name,
                "animal": animal,
                "id": animal_id,
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

    print("\nGenerating full-dimension embeddings ...")
    file_paths = [str(f) for f in image_files]
    full_embeddings = model.encode(file_paths, show_progress_bar=True)
    full_dims = full_embeddings.shape[1]
    print(f"Full embedding dimension: {full_dims}")

    max_needed = max(MATRYOSHKA_DIMS)
    if full_dims < max_needed:
        print(f"Warning: model outputs {full_dims} dims, but {max_needed} requested. "
              f"Skipping dimensions larger than {full_dims}.")

    print(f"\nConnecting to Elasticsearch at {ES_URL} ...")
    es = Elasticsearch(ES_URL, api_key=ES_API_KEY, verify_certs=False, ssl_show_warn=False)
    info = es.info()
    print(f"Connected to cluster: {info['cluster_name']}")

    for dims in MATRYOSHKA_DIMS:
        if dims > full_dims:
            print(f"\nSkipping {dims}-dim index (model only produces {full_dims} dims)")
            continue

        name = index_name(dims)
        truncated = full_embeddings[:, :dims]

        print(f"\n--- {dims}-dim: index '{name}' ---")
        create_index(es, name, dims)

        success, errors = bulk(es, generate_actions(name, image_files, truncated))
        print(f"Indexed {success} documents successfully.")
        if errors:
            print(f"Errors: {errors}")

    print("\nDone.")


if __name__ == "__main__":
    main()
