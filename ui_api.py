import requests
import streamlit as st


st.set_page_config(page_title="Regulations RAG (API)", layout="wide")
st.title("Regulations RAG (API)")
st.caption("Upload a PDF and ask questions. This UI talks to the FastAPI backend.")

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_MODEL_PATH = r"C:\Users\mahen\llama.cpp\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

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
if uploaded and st.button("Upload PDF", type="primary"):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
    resp = requests.post(
        f"{api_base}/upload",
        files=files,
        params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        timeout=120,
    )
    if resp.ok:
        st.success(f"Loaded: {resp.json()}")
    else:
        st.error(f"Upload failed: {resp.status_code} {resp.text}")

st.divider()
try:
    health = requests.get(f"{api_base}/health", timeout=5).json()
    stats = requests.get(f"{api_base}/stats", timeout=5).json()
    st.caption(f"API health: {health} | stats: {stats}")
except Exception as e:
    st.caption(f"API health: error ({e})")

question = st.text_input("Ask a question about the PDF")
if st.button("Answer"):
    payload = {
        "question": question,
        "top_k": top_k,
        "min_score": min_score,
        "model_path": model_path,
        "n_ctx": int(n_ctx),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    resp = requests.post(f"{api_base}/ask", json=payload, timeout=120)
    if resp.ok:
        data = resp.json()
        st.subheader("Answer")
        st.write(data.get("answer", ""))
        st.subheader("References")
        for h in data.get("references", []):
            st.markdown(f"- p{h['page']} (score {h['score']:.2f}): {h['text']}")
    else:
        st.error(f"Ask failed: {resp.status_code} {resp.text}")
