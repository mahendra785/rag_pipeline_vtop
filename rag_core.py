from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_pages(file_obj) -> List[Dict[str, Any]]:
    reader = PdfReader(file_obj)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
    return chunks


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    chunks = []
    for p in pages:
        parts = chunk_text(p["text"], chunk_size, overlap)
        for idx, part in enumerate(parts, start=1):
            chunks.append(
                {
                    "page": p["page"],
                    "chunk": idx,
                    "text": part,
                }
            )
    return chunks


@lru_cache(maxsize=1)
def _get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def build_index(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    if not texts:
        return {"embedder": embedder, "embeddings": np.zeros((0, 0), dtype=np.float32)}
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    return {"embedder": embedder, "embeddings": embeddings}


def retrieve(query: str, index: Dict[str, Any], chunks: List[Dict[str, Any]], top_k: int = 4):
    embedder = index["embedder"]
    embeddings = index["embeddings"]
    if embeddings.size == 0:
        return []
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(embeddings, q)
    top_idx = np.argsort(scores)[::-1][:top_k]
    hits = []
    for i in top_idx:
        hits.append(
            {
                "score": float(scores[i]),
                "page": chunks[i]["page"],
                "chunk": chunks[i]["chunk"],
                "text": chunks[i]["text"],
            }
        )
    return hits


def format_context(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for h in hits:
        tag = f"[p{h['page']}]"
        lines.append(f"{tag} {h['text']}")
    return "\n".join(lines)


def _truncate_context(context: str, n_ctx: int) -> str:
    max_chars = max(1000, (n_ctx * 4) - 800)
    if len(context) <= max_chars:
        return context
    return context[:max_chars]


@lru_cache(maxsize=2)
def _get_llm(model_path: str, n_ctx: int):
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_batch=128,
    )


def answer_with_llama(
    question: str,
    context: str,
    model_path: str,
    n_ctx: int = 4096,
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    llm = _get_llm(model_path, n_ctx)

    system = (
        "You are a RAG assistant for regulations. "
        "Use ONLY the provided context. "
        "If the answer is not in the context, say: 'Not found in the document.' "
        "Cite sources with page tags like [p3]."
    )

    context = _truncate_context(context, n_ctx)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    output = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = output["choices"][0]["message"]["content"].strip()
    if not text:
        return "Not found in the document."
    if "[p" not in text:
        pages = []
        for line in context.splitlines():
            if line.startswith("[p"):
                tag = line.split("]", 1)[0] + "]"
                if tag not in pages:
                    pages.append(tag)
        if pages:
            text = f"{text}\n\nSources: " + " ".join(pages)
    return text
