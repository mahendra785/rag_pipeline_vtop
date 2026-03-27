import io

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
    pages = load_pdf_pages(io.BytesIO(data))
    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF. If this is a scanned PDF, use OCR first.",
        )
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=chunk_overlap)
    index = build_index(chunks)
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

    hits = retrieve(req.question, STATE["index"], STATE["chunks"], top_k=req.top_k)
    if not hits or hits[0]["score"] < req.min_score:
        return {"answer": "Not found in the document.", "references": hits}

    context = format_context(hits)
    answer = answer_with_llama(
        question=req.question,
        context=context,
        model_path=req.model_path,
        n_ctx=req.n_ctx,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return {"answer": answer, "references": hits}
