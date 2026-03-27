import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag_core import (
    load_pdf_pages,
    chunk_pages,
    build_index,
    retrieve,
    format_context,
    answer_with_llama,
)


app = FastAPI(title="Regulations RAG API")

STATE = {
    "pages": None,
    "chunks": None,
    "index": None,
}


class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    min_score: float = 0.0
    model_path: str
    n_ctx: int = 4096
    temperature: float = 0.2
    max_tokens: int = 512


@app.get("/health")
def health():
    return {"ok": True, "has_index": STATE["index"] is not None}


@app.get("/stats")
def stats():
    pages = STATE["pages"] or []
    chunks = STATE["chunks"] or []
    return {"pages": len(pages), "chunks": len(chunks)}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
):
    data = await file.read()
    try:
        pages = load_pdf_pages(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read PDF: {exc}") from exc

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF. If this is a scanned PDF, use OCR first.",
        )
    try:
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=chunk_overlap)
        index = build_index(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {exc}") from exc

    STATE["pages"] = pages
    STATE["chunks"] = chunks
    STATE["index"] = index
    return {"pages": len(pages), "chunks": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    if STATE["index"] is None:
        return {"answer": "Please upload a PDF first.", "references": []}
    if not req.question.strip():
        return {"answer": "Please enter a question.", "references": []}
    if not req.model_path or not Path(req.model_path).is_file():
        raise HTTPException(status_code=400, detail=f"Model file not found: {req.model_path}")

    try:
        hits = retrieve(req.question, STATE["index"], STATE["chunks"], top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not hits or hits[0]["score"] < req.min_score:
        return {"answer": "Not found in the document.", "references": hits}

    context = format_context(hits)
    try:
        answer = answer_with_llama(
            question=req.question,
            context=context,
            model_path=req.model_path,
            n_ctx=req.n_ctx,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return {"answer": answer, "references": hits}
