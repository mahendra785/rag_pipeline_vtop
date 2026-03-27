from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="Regulations RAG (API)", layout="wide")
st.title("Regulations RAG (API)")
st.caption("Upload a PDF and ask questions. This UI talks to the FastAPI backend.")

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_MODEL_PATH = r"C:\Users\mahen\llama.cpp\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"


def _error_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text

with st.sidebar:
    api_base = st.text_input("API Base URL", value=DEFAULT_API)
    model_path = st.text_input("GGUF model path", value=DEFAULT_MODEL_PATH)
    n_ctx = st.number_input("Context size (n_ctx)", min_value=1024, max_value=16384, value=4096, step=512)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=1024, value=512, step=64)
    chunk_size = st.number_input("Chunk size (chars)", min_value=500, max_value=4000, value=1500, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=4, step=1)
    min_score = st.slider("Min similarity (guard)", 0.0, 1.0, 0.0, 0.01)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if "api_ready" not in st.session_state:
    st.session_state.api_ready = False
    st.session_state.api_stats = {"pages": 0, "chunks": 0}

st.divider()
health = None
try:
    health = requests.get(f"{api_base}/health", timeout=5).json()
    stats = requests.get(f"{api_base}/stats", timeout=5).json()
    st.session_state.api_ready = bool(health.get("ok"))
    st.session_state.api_stats = stats
    st.caption(f"API health: {health} | stats: {stats}")
except Exception as e:
    st.session_state.api_ready = False
    st.caption(f"API health: error ({e})")

if uploaded and st.button("Upload PDF", type="primary", disabled=not st.session_state.api_ready):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
    try:
        with st.spinner("Uploading PDF and building backend index..."):
            resp = requests.post(
                f"{api_base}/upload",
                files=files,
                params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                timeout=120,
            )
        if resp.ok:
            data = resp.json()
            st.session_state.api_stats = data
            st.success(f"Loaded {data['pages']} pages and {data['chunks']} chunks.")
        else:
            st.error(f"Upload failed: {_error_detail(resp)}")
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")

if not st.session_state.api_ready:
    st.warning("The FastAPI backend is not reachable. Start `uvicorn api:app --host 127.0.0.1 --port 8000` first.")
elif not Path(model_path).is_file():
    st.warning("The configured GGUF model file does not exist on this machine. Update the model path before asking a question.")

question = st.text_input("Ask a question about the PDF")
can_ask = st.session_state.api_ready and st.session_state.api_stats.get("chunks", 0) > 0
if st.button("Answer", disabled=not can_ask):
    payload = {
        "question": question,
        "top_k": top_k,
        "min_score": min_score,
        "model_path": model_path,
        "n_ctx": int(n_ctx),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    try:
        with st.spinner("Requesting answer from backend..."):
            resp = requests.post(f"{api_base}/ask", json=payload, timeout=120)
        if resp.ok:
            data = resp.json()
            st.subheader("Answer")
            st.write(data.get("answer", ""))
            st.subheader("References")
            for h in data.get("references", []):
                st.markdown(f"- p{h['page']} (score {h['score']:.2f}): {h['text']}")
        else:
            st.error(f"Ask failed: {_error_detail(resp)}")
    except requests.RequestException as exc:
        st.error(f"Ask failed: {exc}")
