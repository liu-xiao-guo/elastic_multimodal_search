import os
import datetime
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer
import base64
from io import BytesIO
import plotly.graph_objects as go
from sklearn.decomposition import PCA

load_dotenv()

ES_URL = os.environ.get("ES_URL", "")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTE_API_KEY", "")
MUSIC_DIR = Path("music")
ANIMALS_DIR = Path("animals")
IMAGES_DIR = Path("images")
MOVIES_DIR = Path("movies")
RECORDED_DIR = Path("recorded")
MULTIMODAL_DIR = Path("multimodal")
MUSIC_INDEX = "music_embeddings"
ANIMAL_INDEX = "animal_embeddings"
IMAGE_INDEX = "image_embeddings"
MOVIE_INDEX = "movie_embeddings"
BOOK_INDEX = "book_embeddings"
MULTIMODAL_INDEX = "multimodal-embeddings"
MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small"
CLIP_MODEL_NAME = "jinaai/jina-clip-v2"
MATRYOSHKA_MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small-retrieval"
EMBEDDINGS_V4_MODEL = "jinaai/jina-embeddings-v4"
MATRYOSHKA_DIMS = [128, 256, 512, 1024]
RAG_MODEL = "openai/gpt-4o-mini"
GEMINI_RAG_MODEL = "google/gemini-3-flash-preview"
GEMINI_API_KEY = os.environ.get("GEMINI_FLASH_API_KEY", "")
TOP_K = 5
MM_NUM_FRAMES = 8


@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    return SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)


@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip_model():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(CLIP_MODEL_NAME, trust_remote_code=True)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading Matryoshka model…")
def load_matryoshka_model() -> SentenceTransformer:
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    return SentenceTransformer(MATRYOSHKA_MODEL_NAME, trust_remote_code=True, device=device)


@st.cache_resource(show_spinner="Loading jina-embeddings-v4 model…")
def load_v4_model() -> SentenceTransformer:
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    return SentenceTransformer(EMBEDDINGS_V4_MODEL, trust_remote_code=True, device=device)


def _smart_textpos_2d(coords: np.ndarray) -> list[str]:
    """Place each label away from its neighbours using inverse-square repulsion."""
    angle_to_pos = [
        ( -22.5,   22.5, "middle right"),
        (  22.5,   67.5, "top right"),
        (  67.5,  112.5, "top center"),
        ( 112.5,  157.5, "top left"),
        ( 157.5,  180.1, "middle left"),
        (-180.1, -157.5, "middle left"),
        (-157.5, -112.5, "bottom left"),
        (-112.5,  -67.5, "bottom center"),
        ( -67.5,  -22.5, "bottom right"),
    ]
    positions = []
    for i, pt in enumerate(coords):
        repulsion = np.zeros(2)
        for j, other in enumerate(coords):
            if i != j:
                diff = pt - other
                d = np.linalg.norm(diff)
                if d > 0:
                    repulsion += diff / d ** 2
        angle = np.degrees(np.arctan2(repulsion[1], repulsion[0]))
        pos = "top center"
        for lo, hi, label in angle_to_pos:
            if lo <= angle < hi:
                pos = label
                break
        positions.append(pos)
    return positions


@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper_model():
    import whisper as _whisper
    return _whisper.load_model("base")


@st.cache_resource(show_spinner=False)
def get_es() -> Elasticsearch:
    return Elasticsearch(
        ES_URL,
        api_key=ES_API_KEY,
        verify_certs=False,
        ssl_show_warn=False,
    )


def save_recording(audio_bytes: bytes) -> Path:
    RECORDED_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RECORDED_DIR / f"recording_{ts}.wav"
    path.write_bytes(audio_bytes)
    return path


def search(embedding, index_name: str, source_fields: list[str]) -> list[dict]:
    es = get_es()
    resp = es.search(
        index=index_name,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": embedding.tolist(),
                "k": TOP_K,
                "num_candidates": 50,
            },
            "_source": source_fields,
        },
    )
    return resp["hits"]["hits"]


_HIGHLIGHT_FIELD = {
    "pre_tags": ["<mark style='background:#FFE066;padding:0 2px;border-radius:3px;'>"],
    "post_tags": ["</mark>"],
    "fragment_size": 300,
    "number_of_fragments": 3,
}


def _make_highlight(query: str | None = None) -> dict:
    field_config = dict(_HIGHLIGHT_FIELD)
    if query:
        field_config["highlight_query"] = {"match": {"source": {"query": query, "analyzer": "english"}}}
    return {"fields": {"source": field_config}}


def book_search(
    query: str,
    embedding,
    search_type: str = "hybrid",
    method: str = "rrf",
    top_k: int = TOP_K,
    knn_boost: float = 0.5,
    use_english_analyzer: bool = False,
) -> list[dict]:
    es = get_es()
    knn = None if embedding is None else {
        "field": "inference_field",
        "query_vector": embedding.tolist(),
        "k": top_k,
        "num_candidates": top_k * 10,
    }
    text_field = "english_source" if use_english_analyzer else "source"

    highlight = _make_highlight(query if use_english_analyzer else None)

    if search_type == "semantic":
        body = {
            "knn": knn,
            "highlight": highlight,
            "_source": ["filename", "source"],
            "size": top_k,
        }
    elif search_type == "lexical":
        body = {
            "query": {"match": {text_field: {"query": query}}},
            "highlight": highlight,
            "_source": ["filename", "source"],
            "size": top_k,
        }
    elif method == "rrf":
        body = {
            "query": {"match": {text_field: {"query": query}}},
            "knn": knn,
            "rank": {"rrf": {"rank_window_size": top_k * 10, "rank_constant": 60}},
            "highlight": highlight,
            "_source": ["filename", "source"],
            "size": top_k,
        }
    else:  # hybrid linear
        bm25_boost = round(1.0 - knn_boost, 4)
        body = {
            "query": {"match": {text_field: {"query": query, "boost": bm25_boost}}},
            "knn": {**(knn or {}), "boost": knn_boost},
            "highlight": highlight,
            "_source": ["filename", "source"],
            "size": top_k,
        }

    return es.search(index=BOOK_INDEX, body=body)["hits"]["hits"]


RERANKER_INFERENCE_ID = ".jina-reranker-v3"


def rerank_results(query: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return hits
    es = get_es()
    passages = [h["_source"].get("source", "") for h in hits]
    response = es.inference.inference(
        inference_id=RERANKER_INFERENCE_ID,
        task_type="rerank",
        body={"query": query, "input": passages},
    )
    reranked_hits = []
    for item in response["rerank"]:
        hit = dict(hits[item["index"]])
        hit["_rerank_score"] = item.get("relevance_score", item.get("score", 0.0))
        reranked_hits.append(hit)
    return reranked_hits


def rerank_multimodal_results(query: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return hits
    es = get_es()
    passages = []
    for h in hits:
        src = h["_source"]
        if src.get("media_type") == "text":
            passages.append(src.get("content", src.get("filename", "")))
        else:
            passages.append(src.get("filename", ""))
    response = es.inference.inference(
        inference_id=RERANKER_INFERENCE_ID,
        task_type="rerank",
        body={"query": query, "input": passages},
    )
    reranked_hits = []
    for item in response["rerank"]:
        hit = dict(hits[item["index"]])
        hit["_rerank_score"] = item.get("relevance_score", item.get("score", 0.0))
        reranked_hits.append(hit)
    return reranked_hits


def generate_rag_answer(query: str, hits: list[dict]) -> str:
    context_parts = []
    for i, hit in enumerate(hits, start=1):
        passage = hit["_source"].get("source", "")
        context_parts.append(f"[{i}] {passage}")
    context = "\n\n".join(context_parts)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question based solely "
                    "on the provided context passages. Cite passage numbers where relevant. "
                    "If the answer is not in the context, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    return response.choices[0].message.content


# ── Multimodal helpers ───────────────────────────────────────────────────────

def embed_video_query(video_path: str, model: SentenceTransformer) -> np.ndarray:
    """Sample MM_NUM_FRAMES from a video and return the averaged embedding."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), MM_NUM_FRAMES, dtype=int)
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
    if not tmp_paths:
        return model.encode("", task='retrieval')
    frame_embs = model.encode(tmp_paths, task='retrieval', show_progress_bar=False)
    for p in tmp_paths:
        Path(p).unlink(missing_ok=True)
    return frame_embs.mean(axis=0)


def _video_key_frame(video_path: Path) -> Image.Image | None:
    """Extract the middle frame of a video as a PIL Image."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, frame)
        img = Image.open(tmp.name).copy()
        Path(tmp.name).unlink(missing_ok=True)
        return img
    except Exception:
        return None


def _pil_to_base64(image: Image.Image, fmt: str = "jpeg") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt};base64,{b64}"


def generate_multimodal_rag_answer(query_input: str, query_type: str, hits: list[dict]) -> str:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    content_parts: list[dict] = []
    text_context = ""

    # Include the query itself as the opening context
    if query_type == "text":
        text_context = "Based on the following retrieved context:\n"
    elif query_type == "image":
        fp = Path(query_input)
        if fp.exists():
            try:
                img = Image.open(fp).convert("RGB")
                content_parts.append({"type": "image_url", "image_url": {"url": _pil_to_base64(img)}})
            except Exception:
                pass
        text_context = "The user searched using the above image. Based on the following retrieved context:\n"
    elif query_type == "video":
        frame = _video_key_frame(Path(query_input))
        if frame:
            content_parts.append({"type": "image_url", "image_url": {"url": _pil_to_base64(frame)}})
        text_context = "The user searched using the above video (key frame shown). Based on the following retrieved context:\n"

    # Add each retrieved result
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        media_type = src.get("media_type", "text")
        file_path = src.get("file_path", "")
        filename = src.get("filename", "?")
        score = hit.get("_score", 0)

        text_context += f"\n[{i}] {media_type.upper()} — {filename} (score: {score:.3f})\n"

        if media_type == "text":
            text_context += src.get("content", "") + "\n"
        elif media_type == "image":
            fp = Path(file_path)
            if fp.exists():
                try:
                    img = Image.open(fp).convert("RGB")
                    content_parts.append({"type": "image_url", "image_url": {"url": _pil_to_base64(img)}})
                    text_context += "(image shown above)\n"
                except Exception:
                    text_context += f"[Image: {filename}]\n"
            else:
                text_context += f"[Image not found: {filename}]\n"
        elif media_type == "video":
            fp = Path(file_path)
            if fp.exists():
                frame = _video_key_frame(fp)
                if frame:
                    content_parts.append({"type": "image_url", "image_url": {"url": _pil_to_base64(frame)}})
                    text_context += "(key frame shown above)\n"
                else:
                    text_context += f"[Video: {filename}]\n"
            else:
                text_context += f"[Video not found: {filename}]\n"

    if query_type == "text":
        text_context += f"\nAnswer the question: {query_input}"
    else:
        text_context += "\nDescribe what these results have in common and how they relate to the search query."

    content_parts.insert(0, {"type": "text", "text": text_context})

    response = client.chat.completions.create(
        model=GEMINI_RAG_MODEL,
        messages=[{"role": "user", "content": content_parts}],  # type: ignore[arg-type]
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def generate_image_rag_answer(query: str, hits: list[dict]) -> str:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    content_parts: list[dict] = []
    text_context = (
        f"The user searched for: \"{query}\"\n\n"
        "Based on the following retrieved images:\n"
    )

    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        filename = src.get("filename", "?")
        category = src.get("category", "unknown")
        score = hit.get("_score", 0)
        text_context += f"\n[{i}] {filename} (category: {category}, score: {score:.3f})\n"
        fp = IMAGES_DIR / filename
        if fp.exists():
            try:
                img = Image.open(fp).convert("RGB")
                content_parts.append({"type": "image_url", "image_url": {"url": _pil_to_base64(img)}})
                text_context += "(image shown above)\n"
            except Exception:
                text_context += "[Image could not be loaded]\n"
        else:
            text_context += f"[Image not found: {filename}]\n"

    text_context += (
        "\nDescribe what these images show and how they relate to the search query. "
        "Be concise and specific."
    )
    content_parts.insert(0, {"type": "text", "text": text_context})

    response = client.chat.completions.create(
        model=GEMINI_RAG_MODEL,
        messages=[{"role": "user", "content": content_parts}],  # type: ignore[arg-type]
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal Search",
    page_icon="logo/elastic.png",
    layout="wide",
)

components.html("""
<script>
    const doc = window.parent.document;
    if (!doc._exclusiveMediaListenerAdded) {
        doc.addEventListener('play', function(e) {
            doc.querySelectorAll('audio, video').forEach(function(el) {
                if (el !== e.target) el.pause();
            });
        }, true);
        doc._exclusiveMediaListenerAdded = true;
    }
</script>
""", height=0)

st.title("🔍 Multimodal Search")
st.caption(
    "Powered by **jina-embeddings-v5-omni-small** (text, image, audio & video embeddings) · "
    "**jina-clip-v2** (image search index) · "
    "**Whisper base** (speech-to-text for RAG)."
)

# ── Mode selector ────────────────────────────────────────────────────────────
mode = st.radio(
    "Search mode",
    ["🎵 Music", "🐾 Animals", "🖼️ Images", "🎬 Movies", "📚 Books", "🌐 Multimodal", "🪆 Matryoshka", "🔤 Embeddings"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MUSIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🎵 Music":
    st.subheader("🎵 Music Search")
    st.caption(
        "Record a clip or pick an existing file, then find the closest matches "
        "using semantic vector search."
    )

    for _k, _v in [
        ("music_input_tab", "🎙️ Record"),
        ("music_recorder_key", 0),
        ("music_query_path", None),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Horizontal radio acts as the tab switcher — its return value is detectable on change.
    music_tab = st.radio(
        "Input method",
        ["🎙️ Record", "📂 Choose existing file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Tab switched → clear results and reset the newly active input widget.
    if music_tab != st.session_state.music_input_tab:
        st.session_state.music_query_path = None
        if music_tab == "🎙️ Record":
            st.session_state.music_recorder_key += 1  # fresh audio input
        st.session_state.music_input_tab = music_tab

    st.divider()

    if music_tab == "🎙️ Record":
        audio_value = st.audio_input(
            "Record audio",
            key=f"music_recorder_{st.session_state.music_recorder_key}",
        )
        if audio_value is not None:
            audio_bytes = audio_value.read()
            saved_path = save_recording(audio_bytes)
            st.success(f"Recording saved → `{saved_path}`")
            st.audio(audio_bytes, format="audio/wav")
            st.session_state.music_query_path = str(saved_path)

            if st.button("🎙️ Record Again"):
                st.session_state.music_recorder_key += 1
                st.session_state.music_query_path = None
                st.rerun()

    else:  # Choose existing file
        wav_files = sorted(RECORDED_DIR.glob("*.wav")) if RECORDED_DIR.exists() else []
        if wav_files:
            selected_name = st.selectbox(
                "Select a recorded file",
                options=[f.name for f in wav_files],
            )
            selected_path = RECORDED_DIR / selected_name
            st.audio(str(selected_path))

            if st.button("🔍 Search with this file"):
                st.session_state.music_query_path = str(selected_path)
        else:
            st.info("No recordings yet. Use the **Record** tab to create one.")

    if st.session_state.music_query_path:
        with st.spinner("Generating embedding…"):
            model = load_model()
            embedding = model.encode(st.session_state.music_query_path, task='retrieval')

        with st.spinner(f"Searching for top {TOP_K} matches…"):
            hits = search(embedding, MUSIC_INDEX, ["filename", "song", "style"])

        st.divider()
        st.subheader(f"Top {len(hits)} Matches")

        for rank, hit in enumerate(hits, start=1):
            src = hit["_source"]
            score = hit["_score"]
            audio_path = MUSIC_DIR / src["filename"]

            song_label = src["song"].replace("_", " ").title()
            style_label = src["style"].replace("-", " ").title()

            with st.container(border=True):
                col_info, col_score = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**#{rank} &nbsp; {song_label}**")
                    st.caption(f"Style: {style_label} &nbsp;·&nbsp; File: `{src['filename']}`")
                with col_score:
                    st.markdown(f"<span style='font-size:0.8em;color:grey;'>Score</span><br><span style='font-size:1.1em;font-weight:600;'>{score:.4f}</span>", unsafe_allow_html=True)

                if audio_path.exists():
                    st.audio(str(audio_path))
                else:
                    st.warning(f"Audio file not found: `{audio_path}`")

# ══════════════════════════════════════════════════════════════════════════════
# ANIMAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🐾 Animals":
    st.subheader("🐾 Animal Search")
    st.caption(
        "Upload an animal photo **or** record an animal sound to find the closest matches."
    )

    for _k, _v in [
        ("animal_input_tab", "🖼️ Upload Picture"),
        ("animal_recorder_key", 0),
        ("animal_uploader_key", 0),
        ("animal_query_path", None),
        ("animal_text_query", None),
        ("animal_last_upload_name", None),
        ("animal_embedding_cache", (None, None)),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    animal_tab = st.radio(
        "Input method",
        ["🖼️ Upload Picture", "🎙️ Record Animal Sound", "📝 Text Search"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Tab switched → clear results and reset the newly active input widget.
    if animal_tab != st.session_state.animal_input_tab:
        st.session_state.animal_query_path = None
        st.session_state.animal_text_query = None
        st.session_state.animal_last_upload_name = None
        if animal_tab == "🖼️ Upload Picture":
            st.session_state.animal_uploader_key += 1  # fresh file uploader
        if animal_tab == "🎙️ Record Animal Sound":
            st.session_state.animal_recorder_key += 1  # fresh audio input
        st.session_state.animal_input_tab = animal_tab

    st.divider()

    if animal_tab == "🖼️ Upload Picture":
        uploaded = st.file_uploader(
            "Upload an animal image",
            type=["png", "jpg", "jpeg"],
            key=f"animal_uploader_{st.session_state.animal_uploader_key}",
        )
        if uploaded is not None:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
            if uploaded.name != st.session_state.animal_last_upload_name:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    st.session_state.animal_query_path = str(Path(tmp.name))
                st.session_state.animal_last_upload_name = uploaded.name
        elif st.session_state.animal_last_upload_name is not None:
            # User removed the uploaded file
            st.session_state.animal_last_upload_name = None
            st.session_state.animal_query_path = None

    elif animal_tab == "🎙️ Record Animal Sound":
        animal_audio = st.audio_input(
            "Record an animal sound",
            key=f"animal_recorder_{st.session_state.animal_recorder_key}",
        )
        if animal_audio is not None:
            audio_bytes = animal_audio.read()
            saved_path = save_recording(audio_bytes)
            st.success(f"Recording saved → `{saved_path}`")
            st.audio(audio_bytes, format="audio/wav")
            st.session_state.animal_query_path = str(saved_path)

            if st.button("🎙️ Record Again", key="animal_record_again"):
                st.session_state.animal_recorder_key += 1
                st.session_state.animal_query_path = None
                st.rerun()

    else:  # Text Search
        with st.form("animal_text_search_form"):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                text_input = st.text_input(
                    "Search term",
                    placeholder="e.g. cat, wolf, dog running…",
                    label_visibility="collapsed",
                )
            with col_btn:
                search_clicked = st.form_submit_button("🔍 Search", use_container_width=True)

        if search_clicked:
            query = text_input.strip()
            if query:
                st.session_state.animal_text_query = query
            else:
                st.session_state.animal_text_query = None
                st.warning("Please enter a search term.")

    # Resolve the active query — either a file path or a text string.
    active_query = st.session_state.animal_text_query or st.session_state.animal_query_path

    # Results — driven by session state; embedding cached to avoid re-encoding the same query.
    if active_query:
        cached_path, cached_emb = st.session_state.animal_embedding_cache

        if active_query != cached_path:
            with st.spinner("Generating embedding…"):
                model = load_model()
                cached_emb = model.encode(active_query, task='retrieval')
                st.session_state.animal_embedding_cache = (active_query, cached_emb)

        with st.spinner(f"Searching for top {TOP_K} matches…"):
            hits = search(cached_emb, ANIMAL_INDEX, ["filename", "animal", "id"])

        st.divider()
        st.subheader(f"Top {len(hits)} Matches")

        cols = st.columns(min(len(hits), 5))
        for rank, (hit, col) in enumerate(zip(hits, cols), start=1):
            src = hit["_source"]
            score = hit["_score"]
            image_path = ANIMALS_DIR / src["filename"]

            with col:
                with st.container(border=True):
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.warning("Image not found")
                    st.markdown(f"**#{rank} {src['animal'].title()}**")
                    st.caption(f"`{src['filename']}`")
                    st.markdown(f"<span style='font-size:0.8em;color:grey;'>Score</span><br><span style='font-size:1em;font-weight:600;'>{score:.4f}</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🖼️ Images":
    st.subheader("🖼️ Image Search")
    st.caption(
        "Upload an image **or** type a text query to find the closest matching images "
        "using CLIP-based semantic search."
    )

    for _k, _v in [
        ("image_input_tab", "📝 Text Search"),
        ("image_uploader_key", 0),
        ("image_query_path", None),
        ("image_text_query", None),
        ("image_last_upload_name", None),
        ("image_embedding_cache", (None, None)),
        ("image_use_rag", False),
        ("image_hits", None),
        ("image_rag_answer", None),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    image_tab = st.radio(
        "Input method",
        ["📝 Text Search", "🖼️ Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if image_tab != st.session_state.image_input_tab:
        st.session_state.image_query_path = None
        st.session_state.image_text_query = None
        st.session_state.image_last_upload_name = None
        st.session_state.image_hits = None
        st.session_state.image_rag_answer = None
        if image_tab == "🖼️ Upload Image":
            st.session_state.image_uploader_key += 1
        st.session_state.image_input_tab = image_tab

    st.divider()

    if image_tab == "📝 Text Search":
        with st.form("image_text_search_form"):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                text_input = st.text_input(
                    "Search term",
                    placeholder="e.g. red rose, mountain sunset, sports car…",
                    label_visibility="collapsed",
                )
            with col_btn:
                search_clicked = st.form_submit_button("🔍 Search", use_container_width=True)

        if search_clicked:
            query = text_input.strip()
            if query:
                st.session_state.image_text_query = query
                st.session_state.image_hits = None
                st.session_state.image_rag_answer = None
            else:
                st.session_state.image_text_query = None
                st.warning("Please enter a search term.")

    else:  # Upload Image
        uploaded = st.file_uploader(
            "Upload an image to find visually similar images",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"image_uploader_{st.session_state.image_uploader_key}",
        )
        if uploaded is not None:
            col_prev, _ = st.columns([2, 1])
            with col_prev:
                st.image(uploaded, caption=uploaded.name, use_container_width=True)
            if uploaded.name != st.session_state.image_last_upload_name:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    st.session_state.image_query_path = str(Path(tmp.name))
                st.session_state.image_last_upload_name = uploaded.name
                st.session_state.image_hits = None
                st.session_state.image_rag_answer = None
        elif st.session_state.image_last_upload_name is not None:
            st.session_state.image_last_upload_name = None
            st.session_state.image_query_path = None
            st.session_state.image_hits = None

    use_rag = st.checkbox(
        "🤖 Use Gemini Flash RAG",
        help="Generate a description of the results using Gemini Flash vision.",
    )
    if not use_rag:
        st.session_state.image_rag_answer = None

    active_query = st.session_state.image_text_query or st.session_state.image_query_path

    if active_query:
        cached_key, cached_emb = st.session_state.image_embedding_cache

        if active_query != cached_key:
            with st.spinner("Generating CLIP embedding…"):
                clip_model = load_clip_model()
                if st.session_state.image_query_path and active_query == st.session_state.image_query_path:
                    query_img = Image.open(active_query).convert("RGB")
                    cached_emb = np.asarray(clip_model.encode_image([query_img]))[0]
                else:
                    cached_emb = np.asarray(clip_model.encode_text([active_query]))[0]
                st.session_state.image_embedding_cache = (active_query, cached_emb)

        if st.session_state.image_hits is None:
            with st.spinner(f"Searching for top {TOP_K} matches…"):
                st.session_state.image_hits = search(cached_emb, IMAGE_INDEX, ["filename", "category", "id"])

        hits = st.session_state.image_hits

        st.divider()

        if use_rag and hits:
            if st.session_state.image_rag_answer is None:
                query_label = st.session_state.image_text_query or Path(st.session_state.image_query_path or "").name
                if not OPENROUTER_API_KEY:
                    st.error("OPENROUTE_API_KEY is not set in .env — cannot use RAG.")
                else:
                    with st.spinner("Asking Gemini Flash…"):
                        st.session_state.image_rag_answer = generate_image_rag_answer(query_label, hits)

            if st.session_state.image_rag_answer:
                st.subheader("🤖 Gemini Flash RAG Answer")
                st.markdown(st.session_state.image_rag_answer)
                st.divider()

        st.subheader(f"Top {len(hits)} Matches")

        cols = st.columns(min(len(hits), 5))
        for rank, (hit, col) in enumerate(zip(hits, cols), start=1):
            src = hit["_source"]
            score = hit["_score"]
            image_path = IMAGES_DIR / src["filename"]

            with col:
                with st.container(border=True):
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.warning("Image not found")
                    st.markdown(f"**#{rank} {src['category'].title()}**")
                    st.caption(f"`{src['filename']}`")
                    st.markdown(
                        f"<span style='font-size:0.8em;color:grey;'>Score</span><br>"
                        f"<span style='font-size:1em;font-weight:600;'>{score:.4f}</span>",
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# MOVIE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🎬 Movies":
    st.subheader("🎬 Movie Search")
    st.caption(
        "Upload a picture, type a description, or record audio to find the closest matching movie clips."
    )

    for _k, _v in [
        ("movie_input_tab", "🖼️ Upload Picture"),
        ("movie_recorder_key", 0),
        ("movie_uploader_key", 0),
        ("movie_video_uploader_key", 0),
        ("movie_query_path", None),
        ("movie_text_query", None),
        ("movie_last_upload_name", None),
        ("movie_last_video_name", None),
        ("movie_embedding_cache", (None, None)),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    movie_tab = st.radio(
        "Input method",
        ["🖼️ Upload Picture", "📝 Text Search", "🎙️ Record Audio", "🎬 Select Video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Tab switched → clear results and reset the newly active input widget.
    if movie_tab != st.session_state.movie_input_tab:
        st.session_state.movie_query_path = None
        st.session_state.movie_text_query = None
        st.session_state.movie_last_upload_name = None
        st.session_state.movie_last_video_name = None
        if movie_tab == "🖼️ Upload Picture":
            st.session_state.movie_uploader_key += 1
        if movie_tab == "🎙️ Record Audio":
            st.session_state.movie_recorder_key += 1
        if movie_tab == "🎬 Select Video":
            st.session_state.movie_video_uploader_key += 1
        st.session_state.movie_input_tab = movie_tab

    st.divider()

    if movie_tab == "🖼️ Upload Picture":
        uploaded = st.file_uploader(
            "Upload a picture to find matching movie clips",
            type=["png", "jpg", "jpeg"],
            key=f"movie_uploader_{st.session_state.movie_uploader_key}",
        )
        if uploaded is not None:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
            if uploaded.name != st.session_state.movie_last_upload_name:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    st.session_state.movie_query_path = str(Path(tmp.name))
                st.session_state.movie_last_upload_name = uploaded.name
        elif st.session_state.movie_last_upload_name is not None:
            st.session_state.movie_last_upload_name = None
            st.session_state.movie_query_path = None

    elif movie_tab == "📝 Text Search":
        with st.form("movie_text_search_form"):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                text_input = st.text_input(
                    "Search term",
                    placeholder="e.g. lightsaber duel, matrix lobby, space battle…",
                    label_visibility="collapsed",
                )
            with col_btn:
                search_clicked = st.form_submit_button("🔍 Search", use_container_width=True)

        if search_clicked:
            query = text_input.strip()
            if query:
                st.session_state.movie_text_query = query
            else:
                st.session_state.movie_text_query = None
                st.warning("Please enter a search term.")

    elif movie_tab == "🎙️ Record Audio":
        movie_audio = st.audio_input(
            "Record audio to find matching movie clips",
            key=f"movie_recorder_{st.session_state.movie_recorder_key}",
        )
        if movie_audio is not None:
            audio_bytes = movie_audio.read()
            saved_path = save_recording(audio_bytes)
            st.success(f"Recording saved → `{saved_path}`")
            st.audio(audio_bytes, format="audio/wav")
            st.session_state.movie_query_path = str(saved_path)

            if st.button("🎙️ Record Again", key="movie_record_again"):
                st.session_state.movie_recorder_key += 1
                st.session_state.movie_query_path = None
                st.rerun()

    else:  # Select Video
        uploaded_video = st.file_uploader(
            "Upload a video clip to find matching movie clips",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key=f"movie_video_uploader_{st.session_state.movie_video_uploader_key}",
        )
        if uploaded_video is not None:
            st.video(uploaded_video)
            if uploaded_video.name != st.session_state.movie_last_video_name:
                suffix = Path(uploaded_video.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_video.read())
                    st.session_state.movie_query_path = str(Path(tmp.name))
                st.session_state.movie_last_video_name = uploaded_video.name
        elif st.session_state.movie_last_video_name is not None:
            st.session_state.movie_last_video_name = None
            st.session_state.movie_query_path = None

    # Resolve the active query — either a file path or a text string.
    active_query = st.session_state.movie_text_query or st.session_state.movie_query_path

    # Results — embedding cached to avoid re-encoding the same query.
    if active_query:
        cached_path, cached_emb = st.session_state.movie_embedding_cache

        if active_query != cached_path:
            with st.spinner("Generating embedding…"):
                model = load_model()
                cached_emb = model.encode(active_query, task='retrieval')
                st.session_state.movie_embedding_cache = (active_query, cached_emb)

        with st.spinner(f"Searching for top {TOP_K} matches…"):
            hits = search(cached_emb, MOVIE_INDEX, ["filename", "title", "franchise"])

        st.divider()
        st.subheader(f"Top {len(hits)} Matches")

        for rank, hit in enumerate(hits, start=1):
            src = hit["_source"]
            score = hit["_score"]
            video_path = MOVIES_DIR / src["filename"]

            with st.container(border=True):
                col_info, col_score = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**#{rank} &nbsp; {src['title']}**")
                    st.caption(f"Franchise: {src['franchise']} &nbsp;·&nbsp; `{src['filename']}`")
                with col_score:
                    st.markdown(f"<span style='font-size:0.8em;color:grey;'>Score</span><br><span style='font-size:1.1em;font-weight:600;'>{score:.4f}</span>", unsafe_allow_html=True)

                if video_path.exists():
                    st.video(str(video_path))
                else:
                    st.warning(f"Video file not found: `{video_path}`")

# ══════════════════════════════════════════════════════════════════════════════
# BOOKS SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📚 Books":
    st.subheader("📚 Books Search")
    st.caption(
        "Semantic or hybrid BM25 + vector search over book passages. "
        "Optionally generate an answer with RAG via OpenRouter."
    )

    for _k, _v in [
        ("book_last_query", None),
        ("book_use_rag", False),
        ("book_use_rerank", True),
        ("book_hits", None),
        ("book_rag_answer", None),
        ("book_embedding_cache", (None, None)),
        ("book_search_type", "hybrid"),
        ("book_hybrid_method", "rrf"),
        ("book_top_k", 5),
        ("book_knn_boost", 0.5),
        ("book_use_english", True),
        ("book_recorder_key", 0),
        ("book_audio_path", None),
        ("book_audio_hash", None),
        ("book_transcription", None),
        ("book_record_searched", False),
        ("book_active_search_type", "hybrid"),
        ("book_record_rag_prev", False),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Search type lives outside the form so it updates immediately and controls layout.
    col_type, col_m, col_b = st.columns([3, 2, 2])
    with col_type:
        search_type = st.radio(
            "Search type",
            ["lexical", "semantic", "hybrid", "record"],
            format_func=lambda x: {
                "hybrid": "Hybrid", "semantic": "Semantic",
                "lexical": "Lexical", "record": "🎙️ Record",
            }[x],
            horizontal=True,
            key="book_search_type_radio",
        )
    with col_m:
        if search_type == "hybrid":
            hybrid_method = st.radio(
                "Hybrid method",
                ["rrf", "linear"],
                format_func=lambda x: "Linear" if x == "linear" else "RRF",
                horizontal=True,
                key="book_hybrid_method_radio",
            )
        else:
            hybrid_method = None
    with col_b:
        if search_type == "hybrid" and hybrid_method == "linear":
            knn_boost = st.slider(
                "KNN boost",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="book_knn_boost_slider",
            )
        else:
            knn_boost = 0.5

    # Clear stale results whenever the search type radio changes.
    if search_type != st.session_state.book_active_search_type:
        st.session_state.book_hits = None
        st.session_state.book_rag_answer = None
        st.session_state.book_last_query = None
        st.session_state.book_audio_path = None
        st.session_state.book_audio_hash = None
        st.session_state.book_transcription = None
        st.session_state.book_record_searched = False
        st.session_state.book_active_search_type = search_type

    st.divider()

    if search_type == "record":
        # ── Record mode ──────────────────────────────────────────────────────
        st.caption(
            "**RAG off**: records → audio embedding → vector search instantly. "
            "**RAG on**: Whisper auto-transcribes → rerank → RAG answer."
        )

        # RAG is the first control; reranking only appears when RAG is on.
        col_rag_r, col_topk_r = st.columns([4, 1])
        with col_rag_r:
            use_rag = st.checkbox(
                "Use RAG — generate an answer from retrieved passages",
                value=False, key="book_record_rag",
            )
        with col_topk_r:
            top_k = st.number_input(
                "Top K", min_value=1, max_value=20, value=5, step=1,
                key="book_top_k_record",
            )

        # Reset everything when RAG is toggled
        if use_rag != st.session_state.book_record_rag_prev:
            st.session_state.book_record_rag_prev = use_rag
            st.session_state.book_recorder_key += 1
            st.session_state.book_audio_path = None
            st.session_state.book_audio_hash = None
            st.session_state.book_transcription = None
            st.session_state.book_last_query = None
            st.session_state.book_hits = None
            st.session_state.book_rag_answer = None
            st.session_state.book_record_searched = False
            st.rerun()

        if use_rag:
            use_rerank = st.checkbox(
                "Enable reranking", value=True, key="book_record_rerank",
                help="Rerank results with .jina-reranker-v3 before feeding the RAG prompt.",
            )
        else:
            use_rerank = False

        book_audio = st.audio_input(
            "Record audio",
            key=f"book_recorder_{st.session_state.book_recorder_key}",
        )
        if book_audio is not None:
            import hashlib as _hashlib
            audio_bytes = book_audio.read()
            audio_hash = _hashlib.md5(audio_bytes).hexdigest()[:12]

            # Only save + reset state when it's genuinely a new recording.
            if audio_hash != st.session_state.book_audio_hash:
                saved_path = save_recording(audio_bytes)
                st.session_state.book_audio_path = str(saved_path)
                st.session_state.book_audio_hash = audio_hash
                st.session_state.book_transcription = None
                st.session_state.book_last_query = None
                st.session_state.book_hits = None
                st.session_state.book_rag_answer = None
                st.session_state.book_record_searched = False

            st.audio(audio_bytes, format="audio/wav")

            if st.button("🎙️ Record Again", key="book_record_again"):
                st.session_state.book_recorder_key += 1
                st.session_state.book_audio_path = None
                st.session_state.book_audio_hash = None
                st.session_state.book_transcription = None
                st.session_state.book_last_query = None
                st.session_state.book_hits = None
                st.session_state.book_rag_answer = None
                st.session_state.book_record_searched = False
                st.rerun()

            if use_rag:
                # Auto-transcribe when RAG is on.
                if st.session_state.book_transcription is None:
                    with st.spinner("Transcribing via Whisper (base)…"):
                        try:
                            _wm = load_whisper_model()
                            _tr = _wm.transcribe(st.session_state.book_audio_path)
                            st.session_state.book_transcription = _tr["text"].strip()
                        except Exception as _e:
                            st.error(f"Transcription failed: {_e}")

                if st.session_state.book_transcription is not None:
                    st.markdown("**Transcribed text** — edit if needed, then click Search to re-run:")
                    st.text_area(
                        "Transcription",
                        value=st.session_state.book_transcription,
                        label_visibility="collapsed",
                        height=80,
                        key="book_transcription_edit",
                    )

                    # Auto-search immediately after first transcription.
                    if not st.session_state.book_record_searched:
                        st.session_state.book_last_query = st.session_state.book_transcription or None
                        st.session_state.book_use_rag = True
                        st.session_state.book_use_rerank = use_rerank
                        st.session_state.book_top_k = int(top_k)
                        st.session_state.book_hits = None
                        st.session_state.book_rag_answer = None
                        st.session_state.book_record_searched = True

                    # Keep Search button so the user can re-run after editing.
                    if st.button("🔍 Search", key="book_record_search", type="primary"):
                        rag_text = st.session_state.get("book_transcription_edit", "").strip()
                        st.session_state.book_last_query = rag_text or None
                        st.session_state.book_use_rag = True
                        st.session_state.book_use_rerank = use_rerank
                        st.session_state.book_top_k = int(top_k)
                        st.session_state.book_hits = None
                        st.session_state.book_rag_answer = None
                        st.session_state.book_record_searched = True
            else:
                # RAG off: kick off vector search immediately, no button needed.
                st.session_state.book_last_query = None
                st.session_state.book_use_rag = False
                st.session_state.book_use_rerank = False
                st.session_state.book_top_k = int(top_k)
                st.session_state.book_record_searched = True

        search_clicked = False  # record mode drives search via state, not this flag

    else:
        # ── Text search form ─────────────────────────────────────────────────
        with st.form("book_search_form"):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                text_input = st.text_input(
                    "Search term",
                    placeholder="e.g. rabbit hole, mad hatter, queen of hearts…",
                    label_visibility="collapsed",
                )
            with col_btn:
                search_clicked = st.form_submit_button("🔍 Search", use_container_width=True)

            col_topk, col_rag, col_rerank = st.columns([1, 3, 2])
            with col_topk:
                top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
            with col_rag:
                use_rag = st.checkbox("Use RAG — generate an answer from retrieved passages")
            with col_rerank:
                use_rerank = st.checkbox("Enable reranking", value=True, help="Rerank results using .jina-reranker-v3")
            if search_type in ("hybrid", "lexical"):
                use_english = st.checkbox(
                    "Use English analyzer",
                    value=st.session_state.book_use_english,
                    help="BM25 matches against english_source (stemming + stop-words) instead of source",
                )
            else:
                use_english = False

        if search_clicked:
            query = text_input.strip()
            if query:
                st.session_state.book_last_query = query
                st.session_state.book_use_rag = use_rag
                st.session_state.book_use_rerank = use_rerank
                st.session_state.book_search_type = search_type
                st.session_state.book_hybrid_method = hybrid_method
                st.session_state.book_knn_boost = knn_boost
                st.session_state.book_top_k = int(top_k)
                st.session_state.book_use_english = use_english
                st.session_state.book_hits = None
                st.session_state.book_rag_answer = None
            else:
                st.warning("Please enter a search term.")

    # ── Results ───────────────────────────────────────────────────────────────
    if search_type == "record" and st.session_state.book_record_searched and st.session_state.book_audio_path:
        # Search uses the AUDIO embedding; RAG / reranking use the transcribed text.
        audio_path = st.session_state.book_audio_path
        rag_query = st.session_state.book_last_query  # may be None if transcription was empty

        if st.session_state.book_hits is None:
            cached_key, cached_emb = st.session_state.book_embedding_cache
            if audio_path != cached_key:
                with st.spinner("Generating audio embedding…"):
                    model = load_model()
                    cached_emb = model.encode(audio_path, task='retrieval')
                    st.session_state.book_embedding_cache = (audio_path, cached_emb)

            with st.spinner(f"Searching for top {st.session_state.book_top_k} matches…"):
                hits = book_search(
                    query=rag_query or "",
                    embedding=cached_emb,
                    search_type="semantic",
                    top_k=st.session_state.book_top_k,
                )

            if st.session_state.book_use_rerank and hits and rag_query:
                with st.spinner("Reranking results with .jina-reranker-v3…"):
                    hits = rerank_results(rag_query, hits)

            st.session_state.book_hits = hits

        hits = st.session_state.book_hits
        st.divider()

        if st.session_state.book_use_rag and rag_query:
            st.subheader("🤖 RAG Answer")
            st.caption(f"Question from transcription: *\"{rag_query}\"*")
            if st.session_state.book_rag_answer is None:
                with st.spinner("Generating answer via OpenRouter…"):
                    st.session_state.book_rag_answer = generate_rag_answer(rag_query, hits)
            st.markdown(st.session_state.book_rag_answer)
            st.divider()
        elif st.session_state.book_use_rag and not rag_query:
            st.info("RAG skipped — transcription was empty.", icon="ℹ️")

        st.subheader(f"Top {len(hits)} Matches")
        for rank, hit in enumerate(hits, start=1):
            src = hit["_source"]
            score = hit["_score"]
            rerank_score = hit.get("_rerank_score")
            with st.container(border=True):
                col_info, col_score = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**#{rank} &nbsp; `{src['filename']}`**")
                with col_score:
                    if rerank_score is not None:
                        st.markdown(f"<span style='font-size:0.8em;color:grey;'>Rerank Score</span><br><span style='font-size:1.1em;font-weight:600;'>{rerank_score:.4f}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='font-size:0.8em;color:grey;'>Score</span><br><span style='font-size:1.1em;font-weight:600;'>{score:.4f}</span>", unsafe_allow_html=True)
                st.write(src.get("source", ""))

    elif search_type != "record" and st.session_state.book_last_query:
        query = st.session_state.book_last_query

        if st.session_state.book_hits is None:
            cached_query, cached_emb = st.session_state.book_embedding_cache
            if st.session_state.book_search_type != "lexical":
                if query != cached_query:
                    with st.spinner("Generating embedding…"):
                        model = load_model()
                        cached_emb = model.encode(query, task='retrieval')
                        st.session_state.book_embedding_cache = (query, cached_emb)
            else:
                cached_emb = None

            with st.spinner(f"Searching for top {st.session_state.book_top_k} matches…"):
                hits = book_search(
                    query, cached_emb,
                    search_type=st.session_state.book_search_type,
                    method=st.session_state.book_hybrid_method,
                    top_k=st.session_state.book_top_k,
                    knn_boost=st.session_state.book_knn_boost,
                    use_english_analyzer=st.session_state.book_use_english,
                )

            if st.session_state.book_use_rerank and hits:
                with st.spinner("Reranking results with .jina-reranker-v3…"):
                    hits = rerank_results(query, hits)

            st.session_state.book_hits = hits

        hits = st.session_state.book_hits

        st.divider()

        if st.session_state.book_use_rag:
            # ── RAG answer ────────────────────────────────────────────────────
            st.subheader("🤖 RAG Answer")

            if st.session_state.book_rag_answer is None:
                with st.spinner("Generating answer via OpenRouter…"):
                    st.session_state.book_rag_answer = generate_rag_answer(query, hits)

            st.markdown(st.session_state.book_rag_answer)
            st.divider()

        # ── Top matches (always shown) ────────────────────────────────────────
        st.subheader(f"Top {len(hits)} Matches")

        for rank, hit in enumerate(hits, start=1):
            src = hit["_source"]
            score = hit["_score"]
            rerank_score = hit.get("_rerank_score")
            highlights = hit.get("highlight", {}).get("source", [])

            with st.container(border=True):
                col_info, col_score = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**#{rank} &nbsp; `{src['filename']}`**")
                with col_score:
                    if rerank_score is not None:
                        st.markdown(f"<span style='font-size:0.8em;color:grey;'>Rerank Score</span><br><span style='font-size:1.1em;font-weight:600;'>{rerank_score:.4f}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='font-size:0.8em;color:grey;'>Score</span><br><span style='font-size:1.1em;font-weight:600;'>{score:.4f}</span>", unsafe_allow_html=True)

                if highlights:
                    for fragment in highlights:
                        st.markdown(
                            f"<div style='font-size:0.9em;line-height:1.6;padding:6px 0;'>"
                            f"…{fragment}…</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write(src.get("source", ""))

# ══════════════════════════════════════════════════════════════════════════════
# MULTIMODAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🌐 Multimodal":
    st.subheader("🌐 Multimodal Search")
    st.caption(
        "Search across images, videos, and texts in one unified index. "
        "Input a text query, upload an image, or upload a video clip."
    )

    for _k, _v in [
        ("mm_input_tab", "📝 Text"),
        ("mm_uploader_key", 0),
        ("mm_video_key", 0),
        ("mm_query_input", None),
        ("mm_query_type", None),
        ("mm_last_img_name", None),
        ("mm_last_vid_name", None),
        ("mm_embedding_cache", (None, None, None)),
        ("mm_raw_hits", None),
        ("mm_reranked_hits", None),
        ("mm_rag_answer", None),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    mm_tab = st.radio(
        "Input method",
        ["📝 Text", "🖼️ Upload Image", "🎬 Upload Video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mm_tab != st.session_state.mm_input_tab:
        st.session_state.mm_query_input = None
        st.session_state.mm_query_type = None
        st.session_state.mm_last_img_name = None
        st.session_state.mm_last_vid_name = None
        st.session_state.mm_raw_hits = None
        st.session_state.mm_reranked_hits = None
        st.session_state.mm_rag_answer = None
        if mm_tab == "🖼️ Upload Image":
            st.session_state.mm_uploader_key += 1
        if mm_tab == "🎬 Upload Video":
            st.session_state.mm_video_key += 1
        st.session_state.mm_input_tab = mm_tab

    st.divider()

    if mm_tab == "📝 Text":
        with st.form("mm_text_form"):
            col_in, col_btn = st.columns([5, 1])
            with col_in:
                text_input = st.text_input(
                    "Query",
                    placeholder="e.g. lightsaber duel, May the Force be with you, rainy city…",
                    label_visibility="collapsed",
                )
            with col_btn:
                search_clicked = st.form_submit_button("🔍 Search", use_container_width=True)
        if search_clicked:
            query = text_input.strip()
            if query:
                st.session_state.mm_query_input = query
                st.session_state.mm_query_type = "text"
                st.session_state.mm_raw_hits = None
                st.session_state.mm_reranked_hits = None
                st.session_state.mm_rag_answer = None
            else:
                st.warning("Please enter a search term.")

    elif mm_tab == "🖼️ Upload Image":
        uploaded_img = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"mm_img_{st.session_state.mm_uploader_key}",
        )
        if uploaded_img is not None:
            col_prev, _ = st.columns([2, 1])
            with col_prev:
                st.image(uploaded_img, caption=uploaded_img.name, use_container_width=True)
            if uploaded_img.name != st.session_state.mm_last_img_name:
                suffix = Path(uploaded_img.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_img.read())
                    st.session_state.mm_query_input = tmp.name
                    st.session_state.mm_query_type = "image"
                st.session_state.mm_last_img_name = uploaded_img.name
                st.session_state.mm_raw_hits = None
                st.session_state.mm_reranked_hits = None
                st.session_state.mm_rag_answer = None
        elif st.session_state.mm_last_img_name is not None:
            st.session_state.mm_last_img_name = None
            st.session_state.mm_query_input = None
            st.session_state.mm_query_type = None

    else:  # Upload Video
        uploaded_vid = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key=f"mm_vid_{st.session_state.mm_video_key}",
        )
        if uploaded_vid is not None:
            st.video(uploaded_vid)
            if uploaded_vid.name != st.session_state.mm_last_vid_name:
                suffix = Path(uploaded_vid.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_vid.read())
                    st.session_state.mm_query_input = tmp.name
                    st.session_state.mm_query_type = "video"
                st.session_state.mm_last_vid_name = uploaded_vid.name
                st.session_state.mm_raw_hits = None
                st.session_state.mm_reranked_hits = None
                st.session_state.mm_rag_answer = None
        elif st.session_state.mm_last_vid_name is not None:
            st.session_state.mm_last_vid_name = None
            st.session_state.mm_query_input = None
            st.session_state.mm_query_type = None

    # Options row: reranking + RAG
    col_rerank, col_rag = st.columns(2)
    with col_rerank:
        use_rerank = st.checkbox(
            "🔀 Enable reranking",
            help="Rerank results using .jina-reranker-v3. Most effective for text queries.",
        )
    with col_rag:
        use_rag = st.checkbox(
            "🤖 Use Gemini Flash Multimodal RAG",
            help="Generate an answer using Gemini Flash with multimodal context from the top results.",
        )

    if not use_rag:
        st.session_state.mm_rag_answer = None
    if not use_rerank:
        st.session_state.mm_reranked_hits = None

    # ── Run search ────────────────────────────────────────────────────────────
    mm_query_input = st.session_state.mm_query_input
    mm_query_type = st.session_state.mm_query_type

    if mm_query_input:
        cached_key, cached_type, cached_emb = st.session_state.mm_embedding_cache

        if mm_query_input != cached_key or mm_query_type != cached_type:
            with st.spinner("Generating embedding…"):
                model = load_model()
                if mm_query_type == "video":
                    cached_emb = embed_video_query(mm_query_input, model)
                else:
                    cached_emb = model.encode(mm_query_input, task='retrieval')
                st.session_state.mm_embedding_cache = (mm_query_input, mm_query_type, cached_emb)

        if st.session_state.mm_raw_hits is None:
            with st.spinner(f"Searching {MULTIMODAL_INDEX} for top {TOP_K} matches…"):
                st.session_state.mm_raw_hits = search(
                    cached_emb, MULTIMODAL_INDEX,
                    ["filename", "file_path", "media_type", "content"],
                )

        # Reranking — only meaningful when the query is text
        if use_rerank:
            if mm_query_type != "text":
                st.info("Reranking requires a text query. Showing vector search results instead.")
            elif st.session_state.mm_reranked_hits is None:
                with st.spinner("Reranking results with .jina-reranker-v3…"):
                    st.session_state.mm_reranked_hits = rerank_multimodal_results(
                        mm_query_input, st.session_state.mm_raw_hits
                    )

        hits = (
            st.session_state.mm_reranked_hits
            if use_rerank and mm_query_type == "text" and st.session_state.mm_reranked_hits
            else st.session_state.mm_raw_hits
        )
        st.divider()

        # ── RAG answer ────────────────────────────────────────────────────────
        if use_rag and hits:
            if st.session_state.mm_rag_answer is None:
                if not OPENROUTER_API_KEY:
                    st.error("OPENROUTE_API_KEY is not set in .env — cannot use RAG.")
                else:
                    with st.spinner("Asking Gemini Flash Multimodal…"):
                        st.session_state.mm_rag_answer = generate_multimodal_rag_answer(
                            mm_query_input, mm_query_type or "text", hits
                        )

            if st.session_state.mm_rag_answer:
                st.subheader("🤖 Gemini Flash RAG Answer")
                st.markdown(st.session_state.mm_rag_answer)
                st.divider()

        # ── Results ───────────────────────────────────────────────────────────
        st.subheader(f"Top {len(hits)} Matches")

        TYPE_META = {
            "image": ("🖼️", "#E8F5E9", "#2E7D32"),
            "video": ("🎬", "#E3F2FD", "#1565C0"),
            "text":  ("📝", "#FFF8E1", "#E65100"),
        }

        for rank, hit in enumerate(hits, start=1):
            src = hit["_source"]
            score = hit["_score"]
            rerank_score = hit.get("_rerank_score")
            media_type = src.get("media_type", "text")
            filename = src.get("filename", "?")
            file_path = src.get("file_path", "")
            content = src.get("content", "")
            icon, bg, fg = TYPE_META.get(media_type, ("📄", "#F5F5F5", "#333"))

            with st.container(border=True):
                col_meta, col_score = st.columns([5, 1])
                with col_meta:
                    st.markdown(
                        f"{icon}&nbsp;<span style='background:{bg};color:{fg};padding:2px 9px;"
                        f"border-radius:10px;font-size:0.74em;font-weight:700;letter-spacing:0.05em;'>"
                        f"{media_type.upper()}</span>&nbsp; **#{rank} &nbsp; `{filename}`**",
                        unsafe_allow_html=True,
                    )
                with col_score:
                    if rerank_score is not None:
                        st.markdown(
                            f"<span style='font-size:0.8em;color:grey;'>Rerank</span><br>"
                            f"<span style='font-size:1.1em;font-weight:600;'>{rerank_score:.4f}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<span style='font-size:0.8em;color:grey;'>Score</span><br>"
                            f"<span style='font-size:1.1em;font-weight:600;'>{score:.4f}</span>",
                            unsafe_allow_html=True,
                        )

                if media_type == "image":
                    fp = Path(file_path)
                    if fp.exists():
                        col_img, _ = st.columns([2, 1])
                        with col_img:
                            st.image(str(fp), use_container_width=True)
                    else:
                        st.warning(f"Image not found: `{file_path}`")

                elif media_type == "video":
                    fp = Path(file_path)
                    if fp.exists():
                        st.video(str(fp))
                    else:
                        st.warning(f"Video not found: `{file_path}`")

                elif media_type == "text":
                    st.markdown(
                        f"<div style='font-size:1.05em;line-height:1.75;padding:10px 14px;"
                        f"background:{bg};border-left:4px solid {fg};"
                        f"border-radius:4px;margin-top:6px;'>{content}</div>",
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# MATRYOSHKA SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🪆 Matryoshka":
    st.subheader("🪆 Matryoshka Embeddings Search")
    st.caption(
        "Compare search quality across embedding dimensions (128 → 1024). "
        "Upload a picture or type a query — results are fetched from the index "
        "whose dimension you select."
    )

    # ── Index statistics ──────────────────────────────────────────────────────
    es = get_es()
    stat_cols = st.columns(4)
    for col, dims in zip(stat_cols, MATRYOSHKA_DIMS):
        idx = f"matryoshka_{dims}_embeddings"
        try:
            stats = es.indices.stats(index=idx)
            size_bytes = stats["_all"]["total"]["store"]["size_in_bytes"]
            if size_bytes >= 1_073_741_824:
                label = f"{size_bytes / 1_073_741_824:.2f} GB"
            elif size_bytes >= 1_048_576:
                label = f"{size_bytes / 1_048_576:.2f} MB"
            elif size_bytes >= 1_024:
                label = f"{size_bytes / 1_024:.1f} KB"
            else:
                label = f"{size_bytes} B"
        except Exception:
            label = "index not found"
        with col:
            st.metric(label=f"{dims}-dim index", value=label, help=idx)

    st.divider()

    # ── Session state ─────────────────────────────────────────────────────────
    for _k, _v in [
        ("mry_input_tab", "🖼️ Upload Picture"),
        ("mry_uploader_key", 0),
        ("mry_query_path", None),
        ("mry_text_query", None),
        ("mry_last_upload_name", None),
        ("mry_embedding_cache", (None, None)),
        ("mry_dims", 1024),
        ("mry_hits", None),
        ("mry_search_key", (None, None)),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Dimension selector — bound directly to mry_dims via key ───────────────
    # Changing dimension auto-triggers a rerun; the search section below detects
    # the new (query, dims) key and re-searches without re-encoding.
    st.radio(
        "Embedding dimensions",
        MATRYOSHKA_DIMS,
        format_func=lambda d: f"{d}-dim",
        horizontal=True,
        key="mry_dims",
    )

    st.divider()

    # ── Input tabs ────────────────────────────────────────────────────────────
    mry_tab = st.radio(
        "Input method",
        ["🖼️ Upload Picture", "📝 Text Search"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mry_tab != st.session_state.mry_input_tab:
        st.session_state.mry_query_path = None
        st.session_state.mry_text_query = None
        st.session_state.mry_last_upload_name = None
        st.session_state.mry_hits = None
        st.session_state.mry_search_key = (None, None)
        if mry_tab == "🖼️ Upload Picture":
            st.session_state.mry_uploader_key += 1
        st.session_state.mry_input_tab = mry_tab

    st.divider()

    if mry_tab == "🖼️ Upload Picture":
        uploaded = st.file_uploader(
            "Upload an image to find similar animal pictures",
            type=["png", "jpg", "jpeg"],
            key=f"mry_uploader_{st.session_state.mry_uploader_key}",
        )
        if uploaded is not None:
            col_prev, _ = st.columns([2, 1])
            with col_prev:
                st.image(uploaded, caption=uploaded.name, use_container_width=True)
            if uploaded.name != st.session_state.mry_last_upload_name:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    st.session_state.mry_query_path = str(Path(tmp.name))
                st.session_state.mry_last_upload_name = uploaded.name
                st.session_state.mry_embedding_cache = (None, None)
                st.session_state.mry_hits = None
                st.session_state.mry_search_key = (None, None)
        elif st.session_state.mry_last_upload_name is not None:
            # User removed the uploaded file
            st.session_state.mry_last_upload_name = None
            st.session_state.mry_query_path = None
            st.session_state.mry_hits = None
            st.session_state.mry_search_key = (None, None)

    else:  # Text Search
        with st.form("mry_text_search_form"):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                text_input = st.text_input(
                    "Search term",
                    placeholder="e.g. golden retriever, eagle in flight, striped cat…",
                    label_visibility="collapsed",
                )
            with col_btn:
                search_clicked = st.form_submit_button("🔍 Search", use_container_width=True)

        if search_clicked:
            query = text_input.strip()
            if query:
                st.session_state.mry_text_query = query
                st.session_state.mry_embedding_cache = (None, None)
                st.session_state.mry_hits = None
                st.session_state.mry_search_key = (None, None)
            else:
                st.session_state.mry_text_query = None
                st.session_state.mry_hits = None
                st.session_state.mry_search_key = (None, None)
                st.warning("Please enter a search term.")

    # ── Search & results ──────────────────────────────────────────────────────
    # The search_key is (active_query, dims). Whenever either the query or the
    # selected dimension changes, the key differs from the cached one and a new
    # ES query is issued automatically — no button click required.
    active_query = st.session_state.mry_text_query or st.session_state.mry_query_path

    if active_query:
        dims = st.session_state.mry_dims
        search_key = (active_query, dims)

        # Re-encode only when the query itself changes (not on dimension change).
        cached_key, cached_full_emb = st.session_state.mry_embedding_cache
        if active_query != cached_key:
            with st.spinner("Generating embedding…"):
                mry_model = load_matryoshka_model()
                cached_full_emb = mry_model.encode(active_query, show_progress_bar=False)
                st.session_state.mry_embedding_cache = (active_query, cached_full_emb)

        # Re-search when query OR dims changes.
        if search_key != st.session_state.mry_search_key:
            truncated_emb = cached_full_emb[:dims]
            idx_name = f"matryoshka_{dims}_embeddings"
            with st.spinner(f"Searching `{idx_name}` for top {TOP_K} matches…"):
                st.session_state.mry_hits = search(truncated_emb, idx_name, ["filename", "animal", "id"])
            st.session_state.mry_search_key = search_key

        hits = st.session_state.mry_hits
        if hits:
            st.divider()
            st.subheader(f"Top {len(hits)} Matches — {dims}-dim index")

            cols = st.columns(min(len(hits), 5))
            for rank, (hit, col) in enumerate(zip(hits, cols), start=1):
                src = hit["_source"]
                score = hit["_score"]
                image_path = ANIMALS_DIR / src["filename"]

                with col:
                    with st.container(border=True):
                        if image_path.exists():
                            st.image(str(image_path), use_container_width=True)
                        else:
                            st.warning("Image not found")
                        st.markdown(f"**#{rank} {src['animal'].title()}**")
                        st.caption(f"`{src['filename']}`")
                        st.markdown(
                            f"<span style='font-size:0.8em;color:grey;'>Score</span><br>"
                            f"<span style='font-size:1em;font-weight:600;'>{score:.4f}</span>",
                            unsafe_allow_html=True,
                        )

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🔤 Embeddings":
    st.subheader("🔤 Word Embedding Visualizer")
    st.caption(
        "Enter words to visualise their semantic distances using "
        f"**{EMBEDDINGS_V4_MODEL}**, projected with PCA or t-SNE."
    )

    _EMB_CATEGORY_COLORS = {
        "animals":   "#e74c3c",
        "royalty":   "#f39c12",
        "furniture": "#3498db",
        "food":      "#2ecc71",
        "medical":   "#9b59b6",
        "other":     "#95a5a6",
    }
    _EMB_WORD_CATEGORIES = {
        "cat": "animals", "dog": "animals", "kitten": "animals", "Puppy": "animals","feline": "animals", "canine": "animals",
        "猫": "animals", "狗": "animals",
        "king": "royalty", "queen": "royalty",
        "table": "furniture", "chair": "furniture",
        "pizza": "food", "pasta": "food",
        "asymptomatic": "medical",
    }
    _EMB_DEFAULT_WORDS = [
        "cat","feline", "dog", "kitten", "Puppy", "king", "queen",
        "猫", "狗", "table", "chair", "pizza", "pasta", "asymptomatic",
    ]

    for _k, _v in [
        ("emb_words_text", "\n".join(_EMB_DEFAULT_WORDS)),
        ("emb_embeddings", None),
        ("emb_word_list", None),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Word editor ───────────────────────────────────────────────────────────
    col_words, col_btn = st.columns([3, 1])
    with col_words:
        words_text = st.text_area(
            "Words (one per line)",
            value=st.session_state.emb_words_text,
            height=220,
        )
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        reset_clicked = st.button("↺ Reset to defaults", use_container_width=True)
        generate_clicked = st.button("▶ Generate", use_container_width=True, type="primary")

    # Legend
    st.markdown(
        " &nbsp; ".join(
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{c};margin-right:4px;vertical-align:middle;'></span>{cat.title()}"
            for cat, c in _EMB_CATEGORY_COLORS.items() if cat != "other"
        ) + f" &nbsp; <span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{_EMB_CATEGORY_COLORS['other']};margin-right:4px;vertical-align:middle;'></span>Other",
        unsafe_allow_html=True,
    )

    if reset_clicked:
        st.session_state.emb_words_text = "\n".join(_EMB_DEFAULT_WORDS)
        st.session_state.emb_embeddings = None
        st.session_state.emb_word_list = None
        st.rerun()

    if generate_clicked:
        words = [w.strip() for w in words_text.splitlines() if w.strip()]
        if len(words) < 3:
            st.warning("Please enter at least 3 words.")
        else:
            st.session_state.emb_words_text = "\n".join(words)
            v4_model = load_v4_model()
            with st.spinner(f"Computing embeddings for {len(words)} words…"):
                embs = v4_model.encode(words, task="retrieval", normalize_embeddings=True)
            st.session_state.emb_embeddings = embs
            st.session_state.emb_word_list = words

    # ── Visualisations ────────────────────────────────────────────────────────
    if st.session_state.emb_embeddings is not None:
        words = st.session_state.emb_word_list
        embeddings = st.session_state.emb_embeddings
        n = len(words)
        colors = [_EMB_CATEGORY_COLORS[_EMB_WORD_CATEGORIES.get(w, "other")] for w in words]

        # Cosine similarity matrix (embeddings are normalized, so dot = cosine sim)
        sim_matrix = (embeddings @ embeddings.T).astype(float)
        np.fill_diagonal(sim_matrix, 0.0)
        avg_sim = sim_matrix.sum(axis=1) / max(n - 1, 1)
        np.fill_diagonal(sim_matrix, 1.0)  # restore for heatmap display

        # Bubble size: map avg similarity to marker size range
        _as_min, _as_max = avg_sim.min(), avg_sim.max()
        norm_sim = (avg_sim - _as_min) / max(_as_max - _as_min, 1e-9)
        marker_sz_3d = 4.0 + 7.0 * norm_sim
        marker_sz_2d = 5.0 + 9.0 * norm_sim

        # ── Controls ─────────────────────────────────────────────────────────
        col_proj, col_thr = st.columns([2, 3])
        with col_proj:
            projection = st.radio(
                "Projection",
                ["PCA", "t-SNE"],
                horizontal=True,
                help="t-SNE preserves local neighbourhood structure: similar words cluster together.",
            )
        with col_thr:
            edge_thr = st.slider(
                "Edge similarity threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Only draw edges between pairs whose cosine similarity ≥ this value.",
            )

        # ── Compute coordinates ───────────────────────────────────────────────
        if projection == "PCA":
            n_comp3 = min(3, n)
            pca3 = PCA(n_components=n_comp3)
            c3_raw = pca3.fit_transform(embeddings)
            ev3 = np.append(pca3.explained_variance_ratio_ * 100, np.zeros(3 - n_comp3))
            coords3 = np.hstack([c3_raw, np.zeros((n, 3 - n_comp3))])
            pca2 = PCA(n_components=min(2, n))
            coords2 = pca2.fit_transform(embeddings)
            ev2 = pca2.explained_variance_ratio_ * 100
            ax3 = [f"PC1 ({ev3[0]:.1f}%)", f"PC2 ({ev3[1]:.1f}%)", f"PC3 ({ev3[2]:.1f}%)"]
            ax2 = [f"PC1 ({ev2[0]:.1f}%)", f"PC2 ({ev2[1]:.1f}%)"]
            sub3 = f"Explained variance: {ev3[0]:.1f}% + {ev3[1]:.1f}% + {ev3[2]:.1f}% = {ev3[:3].sum():.1f}%"
            sub2 = f"Explained variance: {ev2[0]:.1f}% + {ev2[1]:.1f}% = {ev2.sum():.1f}%"
        else:
            from sklearn.manifold import TSNE as _TSNE
            _perp = float(max(2, min(n - 1, 5)))
            with st.spinner("Running t-SNE…"):
                coords3 = _TSNE(n_components=3, perplexity=_perp,
                                random_state=42, max_iter=1000).fit_transform(embeddings)
                coords2 = _TSNE(n_components=2, perplexity=_perp,
                                random_state=42, max_iter=1000).fit_transform(embeddings)
            ax3 = ["Dim 1", "Dim 2", "Dim 3"]
            ax2 = ["Dim 1", "Dim 2"]
            sub3 = "t-SNE: nearby points are semantically similar"
            sub2 = "t-SNE: nearby points are semantically similar"

        # ── Inline helpers ────────────────────────────────────────────────────
        def _hex_rgba(hexc, alpha):
            h = hexc.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        import plotly.colors as _pc
        def _sim_color(s):
            return _pc.sample_colorscale("RdYlGn", [max(0.0, min(1.0, float(s)))])[0]

        # group word indices by category for hull rendering
        cat_idx: dict = {}
        for _i, _w in enumerate(words):
            cat_idx.setdefault(_EMB_WORD_CATEGORIES.get(_w, "other"), []).append(_i)

        def _add_hulls_3d(fig):
            for cat, idx_list in cat_idx.items():
                if len(idx_list) < 4:
                    continue
                pts = coords3[idx_list]
                fig.add_trace(go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    alphahull=0, color=_EMB_CATEGORY_COLORS[cat], opacity=0.08,
                    showlegend=False, hoverinfo="skip",
                ))

        def _add_hulls_2d(fig):
            try:
                from scipy.spatial import ConvexHull as _CH
                for cat, idx_list in cat_idx.items():
                    if len(idx_list) < 3:
                        continue
                    pts = coords2[idx_list]
                    try:
                        hull = _CH(pts)
                        hv = hull.vertices
                        hx = list(pts[hv, 0]) + [pts[hv[0], 0]]
                        hy = list(pts[hv, 1]) + [pts[hv[0], 1]]
                        c = _EMB_CATEGORY_COLORS[cat]
                        fig.add_trace(go.Scatter(
                            x=hx, y=hy, mode="lines",
                            fill="toself",
                            fillcolor=_hex_rgba(c, 0.10),
                            line=dict(color=_hex_rgba(c, 0.40), width=1.5, dash="dot"),
                            showlegend=False, hoverinfo="skip",
                        ))
                    except Exception:
                        pass
            except ImportError:
                pass

        def _add_edges_3d(fig):
            for _i in range(n):
                for _j in range(_i + 1, n):
                    _s = float(sim_matrix[_i, _j])
                    if _s < edge_thr:
                        continue
                    fig.add_trace(go.Scatter3d(
                        x=[coords3[_i, 0], coords3[_j, 0]],
                        y=[coords3[_i, 1], coords3[_j, 1]],
                        z=[coords3[_i, 2], coords3[_j, 2]],
                        mode="lines", showlegend=False, opacity=0.60,
                        line=dict(color=_sim_color(_s), width=1.0 + 3.0 * _s),
                        hovertemplate=f"{words[_i]} ↔ {words[_j]}<br>similarity: {_s:.3f}<extra></extra>",
                    ))

        def _add_edges_2d(fig):
            for _i in range(n):
                for _j in range(_i + 1, n):
                    _s = float(sim_matrix[_i, _j])
                    if _s < edge_thr:
                        continue
                    fig.add_trace(go.Scatter(
                        x=[coords2[_i, 0], coords2[_j, 0]],
                        y=[coords2[_i, 1], coords2[_j, 1]],
                        mode="lines", showlegend=False, opacity=0.55,
                        line=dict(color=_sim_color(_s), width=1.0 + 3.0 * _s),
                        hovertemplate=f"{words[_i]} ↔ {words[_j]}<br>similarity: {_s:.3f}<extra></extra>",
                    ))

        tab3d, tab2d, tab_sim = st.tabs(["🌐 3D View", "📊 2D View", "📐 Similarity Matrix"])

        # ── 3D ────────────────────────────────────────────────────────────────
        with tab3d:
            fig3d = go.Figure()
            _add_hulls_3d(fig3d)   # layer 0: category volume hulls
            _add_edges_3d(fig3d)   # layer 1: similarity-filtered coloured edges
            pt_start = len(fig3d.data)   # points start here
            for i, word in enumerate(words):
                fig3d.add_trace(go.Scatter3d(
                    x=[coords3[i, 0]], y=[coords3[i, 1]], z=[coords3[i, 2]],
                    mode="markers+text", name=word, text=[word],
                    textposition="top center",
                    textfont=dict(size=13, color=colors[i]),
                    marker=dict(
                        size=float(marker_sz_3d[i]),
                        color=colors[i], opacity=0.9,
                        line=dict(width=1.5, color="white"),
                    ),
                    hovertemplate=(
                        f"<b>{word}</b><br>"
                        f"avg similarity: {avg_sim[i]:.3f}<br>"
                        f"{ax3[0]}: {coords3[i,0]:.3f}<br>"
                        f"{ax3[1]}: {coords3[i,1]:.3f}<br>"
                        f"{ax3[2]}: {coords3[i,2]:.3f}<extra></extra>"
                    ),
                ))
            pt_idx = list(range(pt_start, pt_start + n))
            fig3d.update_layout(
                title=dict(
                    text=(f"Word Embeddings — 3D {projection}<br>"
                          f"<sup>Model: {EMBEDDINGS_V4_MODEL} | {sub3}</sup>"),
                    x=0.5,
                ),
                scene=dict(
                    xaxis_title=ax3[0], yaxis_title=ax3[1], zaxis_title=ax3[2],
                    xaxis=dict(backgroundcolor="#f8f9fa", gridcolor="white"),
                    yaxis=dict(backgroundcolor="#f0f3fa", gridcolor="white"),
                    zaxis=dict(backgroundcolor="#f5f0fa", gridcolor="white"),
                    camera=dict(eye=dict(x=0.7, y=0.7, z=0.7)),
                ),
                legend=dict(title="Words", x=1.02, y=0.98, xanchor="left", yanchor="top"),
                margin=dict(l=0, r=140, t=80, b=0),
                height=680,
                updatemenus=[dict(
                    type="buttons", showactive=True,
                    x=0.01, y=0.99, xanchor="left", yanchor="top",
                    buttons=[
                        dict(label="Show Labels", method="restyle",
                             args=[{"mode": "markers+text"}, pt_idx]),
                        dict(label="Hide Labels", method="restyle",
                             args=[{"mode": "markers"}, pt_idx]),
                    ],
                )],
            )
            st.plotly_chart(fig3d, use_container_width=True)
            st.caption(
                "**Bubble size** = average cosine similarity to all other words (larger = more central). "
                "**Edge colour**: 🟢 green = high similarity, 🟡 yellow = moderate, 🔴 red = low. "
                "Shaded volumes = category groups (≥ 4 words). "
                "Only edges above the threshold are shown."
            )

        # ── 2D ────────────────────────────────────────────────────────────────
        with tab2d:
            textpos2d = _smart_textpos_2d(coords2)
            fig2d = go.Figure()
            _add_hulls_2d(fig2d)   # layer 0: category convex-hull shading
            _add_edges_2d(fig2d)   # layer 1: similarity-filtered coloured edges
            pt_start_2d = len(fig2d.data)
            for i, word in enumerate(words):
                fig2d.add_trace(go.Scatter(
                    x=[coords2[i, 0]], y=[coords2[i, 1]],
                    mode="markers+text", name=word, text=[word],
                    textposition=textpos2d[i],
                    textfont=dict(size=13, color=colors[i]),
                    marker=dict(
                        size=float(marker_sz_2d[i]),
                        color=colors[i], opacity=0.9,
                        line=dict(width=1.5, color="white"),
                    ),
                    hovertemplate=(
                        f"<b>{word}</b><br>"
                        f"avg similarity: {avg_sim[i]:.3f}<br>"
                        f"{ax2[0]}: {coords2[i,0]:.3f}<br>"
                        f"{ax2[1]}: {coords2[i,1]:.3f}<extra></extra>"
                    ),
                ))
            fig2d.update_layout(
                title=dict(
                    text=(f"Word Embeddings — 2D {projection}<br>"
                          f"<sup>Model: {EMBEDDINGS_V4_MODEL} | {sub2}</sup>"),
                    x=0.5,
                ),
                xaxis_title=ax2[0], yaxis_title=ax2[1],
                xaxis=dict(gridcolor="#ececec", zeroline=True, zerolinecolor="#aaa"),
                yaxis=dict(gridcolor="#ececec", zeroline=True, zerolinecolor="#aaa"),
                plot_bgcolor="#fafafa",
                legend=dict(title="Words", x=1.01, y=1),
                margin=dict(l=60, r=150, t=80, b=60),
                height=640,
            )
            st.plotly_chart(fig2d, use_container_width=True)
            st.caption(
                "**Bubble size** = average cosine similarity to all other words. "
                "**Shaded regions** = category convex hulls (≥ 3 words). "
                "**Edge colour**: 🟢 green = high similarity, 🟡 yellow = moderate, 🔴 red = low."
            )

        # ── Similarity matrix ─────────────────────────────────────────────────
        with tab_sim:
            fig_heat = go.Figure(data=go.Heatmap(
                z=sim_matrix,
                x=words, y=words,
                colorscale="RdYlGn",
                zmin=-1, zmax=1,
                text=[[f"{sim_matrix[i, j]:.3f}" for j in range(n)] for i in range(n)],
                texttemplate="%{text}",
                textfont=dict(size=10),
            ))
            fig_heat.update_layout(
                title=dict(text="Cosine Similarity Matrix", x=0.5),
                height=max(420, n * 42 + 160),
                margin=dict(l=100, r=20, t=60, b=100),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            pairs = [
                (words[i], words[j], float(sim_matrix[i, j]))
                for i in range(n) for j in range(i + 1, n)
            ]
            pairs.sort(key=lambda x: x[2], reverse=True)
            col_most, col_least = st.columns(2)
            with col_most:
                st.markdown("**Most similar pairs**")
                for w1, w2, s in pairs[:5]:
                    st.markdown(f"- `{w1}` ↔ `{w2}`: **{s:.4f}**")
            with col_least:
                st.markdown("**Least similar pairs**")
                for w1, w2, s in pairs[-5:]:
                    st.markdown(f"- `{w1}` ↔ `{w2}`: **{s:.4f}**")
