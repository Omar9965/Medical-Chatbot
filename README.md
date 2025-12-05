# Medical-Chatbot

Professional, privacy-first retrieval-augmented medical question-answering chatbot built with LangChain, HuggingFace embeddings and Pinecone for vector search. This repository contains utilities to load PDF clinical documents, split and embed them, index embeddings in Pinecone, and serve a simple Flask-based chat UI backed by a RAG chain using a Google Generative AI chat model.

## Key Features
- Load PDF documents from a `data/` folder and extract text
- Chunk documents with configurable `chunk_size` and `chunk_overlap` for reliable retrieval
- Generate embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- Store and query embeddings in Pinecone (serverless index)
- Use Google Gemini (via `langchain_google_genai`) as the LLM for final answer generation
- Simple Flask chat UI (`templates/chat.html`) for interactive Q&A

## Repository Structure

```
Medical-Chatbot/
├── app.py                      # Flask app that wires the RAG chain and serves the UI
├── requirements.txt            # Python dependencies
├── src/
│   ├── helper.py               # Document loading, filtering, splitting, embeddings helper
│   └── prompt.py               # System prompt template
├── data/                       # Place your PDF files here
├── templates/
│   └── chat.html               # Frontend chat UI
├── static/
│   └── style.css               # Chat UI styles
├── utils/
│   └── configs.py              # Settings loader (reads .env)
├── Notebook/                   # Development notebook with experiments
└── README.md
```

## Requirements
- Python 3.10+ (tested on 3.12)
- A Python venv is recommended
- See `requirements.txt` for specific package versions

## Environment
Create a `.env` file at the project root with the following keys (example):

```
APP_NAME=Medical-Chatbot
APP_VERSION=1.0
GEMINI_API_KEY=your-google-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

Notes:
- `GEMINI_API_KEY` is used by the `langchain_google_genai` client for chat generation.
- `PINECONE_API_KEY` is used to connect to your Pinecone account and index.

If your `.env` file is stored in the project root (recommended), the `utils.configs.Settings` class will load it automatically. The code also contains a fallback that resolves the `.env` path relative to the project layout.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you run into dependency issues, create a fresh venv and install the packages listed in `requirements.txt`.

## Indexing Documents (one-time)
1. Put your PDFs under the `data/` directory.
2. Run the notebook `Notebook/trials.ipynb` or use the helper functions in `src/helper.py` to:
	- load PDFs (`load_pdf_file`)
	- filter minimal metadata (`filter_to_minimal_docs`)
	- split documents (`split_documents`)
	- create embeddings using `download_embeddings()` (HuggingFace model)
	- push embeddings to Pinecone (the notebook contains example code to create an index and upload vectors)

Important: The Flask app is designed to connect to an existing Pinecone index. It will not re-embed and re-upload your entire corpus on every start (faster and safer in production).

## Run the Flask App

```powershell
# activate venv first
python app.py
```

Open `http://localhost:8080` in your browser to use the chat UI.

## How It Works (high level)
- Documents → chunked → embeddings (HuggingFace) → Pinecone index
- User query → embedding → Pinecone similarity search → retrieved chunks
- Retrieved chunks → system prompt (from `src/prompt.py`) → Chat model (Gemini) → answer

## Tuning Recommendations
- `chunk_size`: smaller values (e.g., 500) increase retrieval precision for dense medical facts; larger values (1000–1500) provide more context per chunk.
- `chunk_overlap`: 10–30% of `chunk_size` is usually helpful to avoid cutting sentences.
- Evaluate retrieval quality and LLM answers iteratively on a development set.

## Troubleshooting
- Module import errors in the notebook: ensure you run the notebook from the repository root or add the project root to `sys.path` (the notebook already inserts the parent directory).
- Missing `.env` keys: confirm `.env` exists at the project root and contains the required keys.
- Pinecone index issues: confirm your Pinecone project, environment, and API key; the notebook includes how to create an index programmatically.

## Security & Privacy Notes
- Do not commit your `.env` or API keys to source control.
- Medical data can be sensitive — ensure you have the right approvals and safeguards before uploading or exposing patient data.

## Contributing
Contributions are welcome. Please open issues or pull requests and include reproducible steps. For major changes open an issue first to discuss.



