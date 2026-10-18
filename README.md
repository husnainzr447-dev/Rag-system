# Company Files RAG Assistant (Python)

A small local RAG project for searching HR policies, employee handbooks, and other approved company files. It uses free local embeddings, FAISS vector search, Streamlit, and Ollama for local answer generation. No paid API key is required.

## How it works

1. Upload PDF, DOCX, or TXT documents.
2. Extract text and split it into overlapping chunks.
3. Embed each chunk with `sentence-transformers/all-MiniLM-L6-v2`.
4. Store vectors in a FAISS index.
5. Retrieve the most relevant chunks for a question.
6. Ask a local Ollama model to answer only from retrieved context.
7. Require login before the app can be used.

## Requirements

- Python 3.10 or 3.11 recommended
- Git (optional, for GitHub)
- Ollama installed locally: https://ollama.com

## Windows PowerShell setup

Open PowerShell in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Configure authentication

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Choose a password and calculate its SHA-256 hash:

```powershell
python -c "import hashlib; print(hashlib.sha256('Choose-A-Strong-Password'.encode()).hexdigest())"
```

Put the printed hash in `.env` as `RAG_PASSWORD_SHA256=...`. Change `RAG_USERNAME` too. Never commit `.env` to GitHub. This is a simple classroom/demo login, not production-grade identity management.

## Start the local answer model

Install Ollama from https://ollama.com, then run:

```powershell
ollama pull llama3.2
```

Ollama normally runs its local service after installation. If needed, open a second terminal and run `ollama serve`.

## Run the app

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`) and sign in with the credentials from `.env`.

## Test checklist

1. **Authentication:** enter a wrong password; access should be denied.
2. **Authentication:** enter the correct username/password; the app should open. Use Log out and confirm it returns to sign-in.
3. **Ingestion:** upload an HR policy PDF or DOCX and click **Save uploaded files**.
4. **Embedding:** click **Build / refresh embeddings** and confirm the number of chunks indexed.
5. **Retrieval:** ask a question whose answer appears in the document. Check the retrieved source filename and text.
6. **Grounding:** ask a question not covered by the documents. The model should say it could not find the answer.
7. **Repeat question:** ask a second question without rebuilding the index; it should use the same index.
8. **Re-indexing:** add/update a file and click **Build / refresh embeddings** again.

## GitHub

Before uploading, check that `.env`, company documents, and `data/index/` are not committed. Do not publish confidential HR policies to a public repository. For a private company repository, follow your organization's access and data-handling rules.

```powershell
git init
git add app.py requirements.txt README.md .env.example .gitignore data/company_docs/.gitkeep
git commit -m "Add company document RAG assistant"
```

If `git` is not recognized on Windows, install Git for Windows from https://git-scm.com/download/win, close and reopen PowerShell, then check `git --version`.

## Important limitations

- This starter uses a basic single shared login. It does not provide per-employee roles, audit logging, password reset, or enterprise SSO.
- Scanned/image-only PDFs need OCR before text can be retrieved.
- Embeddings are local, but the first run downloads the embedding model. Ollama downloads its model separately.
- Treat retrieved HR information as sensitive. Use only approved documents, protect the computer and backups, and ask your IT/security team before deploying this beyond a local demo.
