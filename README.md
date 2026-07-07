# Elastic Multimodal Search

A Streamlit demo application that showcases multimodal semantic search powered by [Elasticsearch](https://www.elastic.co/elasticsearch) and state-of-the-art embedding models.

![Elastic Logo](logo/elastic.png)

---

## Features

The app provides eight search modes, each backed by a dedicated Elasticsearch index:

| Mode | Description | Index |
|---|---|---|
| 🎵 **Music** | Search music by text description or audio recording | `music_embeddings` |
| 🐾 **Animals** | Search animals by image, audio, or text | `animal_embeddings` |
| 🖼️ **Images** | Search images by uploading a picture or typing a description | `image_embeddings` |
| 🎬 **Movies** | Search movie clips by image, text, or audio | `movie_embeddings` |
| 📚 **Books** | Lexical, semantic, hybrid, or audio-recorded search with optional RAG | `book_embeddings` |
| 🌐 **Multimodal** | Cross-modal search across text, image, and audio | `multimodal-embeddings` |
| 🪆 **Matryoshka** | Demonstrate Matryoshka embedding truncation at 128 / 256 / 512 / 1024 dims | `music_embeddings` |
| 🔤 **Embeddings** | Interactive 3D / 2D word embedding visualizer with PCA and t-SNE | *(in-memory)* |

### Search capabilities

- **Lexical search** — BM25 full-text with optional English analyzer (stemming + stop-words)
- **Semantic search** — k-NN vector search against dense embedding fields
- **Hybrid search** — combines BM25 and k-NN via RRF or linear combination
- **Audio recording** — record a clip and search using its audio embedding directly
- **RAG** — retrieval-augmented generation: top-K passages fed to an LLM for a grounded answer
- **Reranking** — results reranked with `jina-reranker-v3` via OpenRouter

---

## Models

| Model | Role |
|---|---|
| `jinaai/jina-embeddings-v5-omni-small` | Text, image, audio & video embeddings (main search) |
| `jinaai/jina-embeddings-v5-omni-small-retrieval` | Matryoshka truncated embeddings |
| `jinaai/jina-clip-v2` | Image search index embeddings |
| `jinaai/jina-embeddings-v4` | Word embedding visualizer |
| `openai-whisper` (base) | Speech-to-text for Books Record + RAG mode |
| `openai/gpt-4o-mini` via OpenRouter | RAG answer generation |
| `google/gemini-3-flash-preview` | Alternative RAG model |

---

## Prerequisites

- Python 3.10+
- A running Elasticsearch 8+ cluster (or [Elastic Cloud](https://cloud.elastic.co))
- An [OpenRouter](https://openrouter.ai) API key (for RAG)
- A Google Gemini API key (optional, for Gemini RAG)

---

## Installation

```bash
git clone <repo-url>
cd elastic_multimodal_search
pip install -r requirements.txt
```

---

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp env.example .env
```

```dotenv
ES_URL="https://<your-cluster>:9200"
ES_API_KEY="<your-elasticsearch-api-key>"
OPENROUTE_API_KEY="<your-openrouter-api-key>"
GEMINI_FLASH_API_KEY="<your-gemini-api-key>"
```

---

## Data Ingestion

Run the ingestion scripts to build each index before launching the app.  
Each script reads from its corresponding data directory and pushes embeddings to Elasticsearch.

```bash
python ingest_audios.py       # music_embeddings  (audio files in music/)
python ingest_animals.py      # animal_embeddings (images/audio in animals/)
python ingest_images.py       # image_embeddings  (images in images/)
python ingest_movies.py       # movie_embeddings  (video frames in movies/)
python ingest_books.py        # book_embeddings   (text passages in books/)
python ingest_multimodal.py   # multimodal-embeddings
python ingest_matryoshka.py   # matryoshka index
```

---

## Running the App

```bash
streamlit run search_app.py
```

The app opens at `http://localhost:8501` by default.

---

## Project Structure

```
elastic_multimodal_search/
├── search_app.py          # Main Streamlit application
├── ingest_audios.py       # Music index ingestion
├── ingest_animals.py      # Animals index ingestion
├── ingest_images.py       # Images index ingestion
├── ingest_movies.py       # Movies index ingestion
├── ingest_books.py        # Books index ingestion
├── ingest_multimodal.py   # Multimodal index ingestion
├── ingest_matryoshka.py   # Matryoshka index ingestion
├── requirements.txt       # Python dependencies
├── env.example            # Environment variable template
├── logo/                  # App logo assets
├── music/                 # Music audio files
├── animals/               # Animal images and audio
├── images/                # Image dataset
├── movies/                # Movie clip frames
├── books/                 # Book text passages
├── multimodal/            # Multimodal dataset
└── recorded/              # Saved audio recordings (auto-created)
```

You can use the images, vidoes inside the directory "searchobj" for trials.

In the directory "demos", it shows some of the search cases.

---

## License

This software is licensed under the Apache License, version 2 ("ALv2").

You may obtain a copy of the License at:

> http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
