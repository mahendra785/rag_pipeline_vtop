import io
import streamlit as st
from rag_core import (
    load_pdf_pages,
    chunk_pages,
    build_index,
    retrieve,
    format_context,
    answer_with_llama,
)


st.set_page_config(page_title="Regulations RAG", layout="wide")

st.title("Regulations RAG (PDF-Only)")
st.caption("Upload a PDF of rules/regulations and ask questions. Answers are grounded only in the PDF with references.")

DEFAULT_MODEL_PATH = r"C:\Users\mahen\llama.cpp\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("GGUF model path", value=DEFAULT_MODEL_PATH)
    n_ctx = st.number_input("Context size (n_ctx)", min_value=1024, max_value=16384, value=4096, step=512)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=1024, value=512, step=64)

    st.header("Retrieval")
    chunk_size = st.number_input("Chunk size (chars)", min_value=500, max_value=4000, value=1500, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=4, step=1)
    min_score = st.slider("Min similarity (guard)", 0.0, 1.0, 0.0, 0.01)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

if uploaded:
    pdf_bytes = uploaded.read()
    pages = load_pdf_pages(io.BytesIO(pdf_bytes))
    if not pages:
        st.error("No extractable text found in PDF. If this is a scanned PDF, use OCR first.")
        st.stop()
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=chunk_overlap)
    index = build_index(chunks)
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.success(f"Loaded {len(pages)} pages, {len(chunks)} chunks.")

question = st.text_input("Ask a question about the PDF")

if st.button("Answer", type="primary") and question:
    if st.session_state.index is None:
        st.warning("Please upload a PDF first.")
    else:
        hits = retrieve(
            question,
            st.session_state.index,
            st.session_state.chunks,
            top_k=top_k,
        )

        if not hits or hits[0]["score"] < min_score:
            st.write("Not found in the document.")
        else:
            context = format_context(hits)
            answer = answer_with_llama(
                question=question,
                context=context,
                model_path=model_path,
                n_ctx=int(n_ctx),
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            st.subheader("Answer")
            st.write(answer)

            st.subheader("References")
            for h in hits:
                st.markdown(
                    f"- p{h['page']} (score {h['score']:.2f}): {h['text']}"
                )
