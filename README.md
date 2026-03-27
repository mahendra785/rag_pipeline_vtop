# Regulations RAG (PDF-Only)

Local, PDF-grounded RAG app using a GGUF model via `llama-cpp-python`.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
streamlit run app.py
```

## Run (API + UI)
Start the API:
```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the API UI:
```powershell
streamlit run ui_api.py
```

## Notes
- Default model path is set in `app.py` but can be edited in the UI.
- Answers are restricted to the PDF context and cite page tags like `[p3]`.
