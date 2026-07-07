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
DATASET_DIR = Path("books")
INDEX_NAME = "book_embeddings"
MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small-retrieval"
CHUNK_SIZE = 1000  # characters per chunk


def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > CHUNK_SIZE and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para) if current else para
    if current:
        chunks.append(current.strip())
    return chunks


def create_index(es: Elasticsearch, dims: int) -> None:
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")

    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "source": {"type": "text", "copy_to": "english_source"},
                "english_source": {"type": "text", "analyzer": "english"},
                "inference_field": {
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


def generate_actions(chunks_by_file: list[tuple[str, list[str]]], embeddings):
    idx = 0
    for filename, chunks in chunks_by_file:
        for chunk in chunks:
            yield {
                "_index": INDEX_NAME,
                "_source": {
                    "filename": filename,
                    "source": chunk,
                    "inference_field": embeddings[idx].tolist(),
                },
            }
            idx += 1


def main():
    if not ES_URL or not ES_API_KEY:
        print("Error: ES_URL and ES_API_KEY environment variables must be set.")
        sys.exit(1)

    text_files = sorted(DATASET_DIR.glob("*.txt"))
    if not text_files:
        print(f"No .txt files found in '{DATASET_DIR}'")
        sys.exit(1)

    print(f"Found {len(text_files)} text files:")
    for f in text_files:
        print(f"  {f.name}")

    print(f"\nLoading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print("\nChunking and generating embeddings ...")
    chunks_by_file = []
    all_chunks = []
    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        chunks_by_file.append((path.name, chunks))
        all_chunks.extend(chunks)
        print(f"  {path.name}: {len(chunks)} chunks")

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    dims = embeddings.shape[1]
    print(f"Embedding dimension: {dims}")

    print(f"\nConnecting to Elasticsearch at {ES_URL} ...")
    es = Elasticsearch(ES_URL, api_key=ES_API_KEY, verify_certs=False, ssl_show_warn=False)
    info = es.info()
    print(f"Connected to cluster: {info['cluster_name']}")

    create_index(es, dims)

    print("\nIndexing documents ...")
    success, errors = bulk(es, generate_actions(chunks_by_file, embeddings))
    print(f"Indexed {success} documents successfully.")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
