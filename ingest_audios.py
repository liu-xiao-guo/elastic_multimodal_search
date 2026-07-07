import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

load_dotenv()

ES_URL = os.environ.get("ES_URL", "")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
DATASET_DIR = Path("music")
INDEX_NAME = "music_embeddings"
MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small-retrieval"

KNOWN_SONGS = ["bella_ciao", "mozart_symphony25"]


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    for song in KNOWN_SONGS:
        if stem.startswith(song + "_"):
            style = stem[len(song) + 1:]
            return song, style
    return "unknown", stem


def create_index(es: Elasticsearch, dims: int) -> None:
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")

    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "song": {"type": "keyword"},
                "style": {"type": "keyword"},
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


def generate_actions(audio_files: list[Path], embeddings):
    for path, emb in zip(audio_files, embeddings):
        song, style = parse_filename(path)
        yield {
            "_index": INDEX_NAME,
            "_source": {
                "filename": path.name,
                "song": song,
                "style": style,
                "embedding": emb.tolist(),
            },
        }


def main():
    if not ES_URL or not ES_API_KEY:
        print("Error: ES_URL and ES_API_KEY environment variables must be set.")
        sys.exit(1)

    audio_files = sorted(DATASET_DIR.glob("*.wav"))
    if not audio_files:
        print(f"No .wav files found in '{DATASET_DIR}'")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"  {f.name}")

    print(f"\nLoading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print("\nGenerating embeddings ...")
    file_paths = [str(f) for f in audio_files]
    embeddings = model.encode(file_paths, show_progress_bar=True)
    dims = embeddings.shape[1]
    print(f"Embedding dimension: {dims}")

    print(f"\nConnecting to Elasticsearch at {ES_URL} ...")
    es = Elasticsearch(ES_URL, api_key=ES_API_KEY, verify_certs=False, ssl_show_warn=False)
    info = es.info()
    print(f"Connected to cluster: {info['cluster_name']}")

    create_index(es, dims)

    print("\nIndexing documents ...")
    success, errors = bulk(es, generate_actions(audio_files, embeddings))
    print(f"Indexed {success} documents successfully.")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
