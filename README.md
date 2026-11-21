# DocuMind AI - AI-Powered Document Intelligence Platform

DocuMind AI is a full-stack document question-answering application that lets users upload documents and ask questions grounded in their uploaded content. The platform combines a React interface, a FastAPI backend, local embeddings, persistent vector search, PostgreSQL, and a language model selected for local development or cloud deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
- [Installation and Setup](#installation-and-setup)
- [Docker Deployment](#docker-deployment)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

## Project Overview

DocuMind AI provides a private, document-focused workspace for retrieving and summarizing information. Registered users can upload PDF, DOCX, or TXT files, search their document list, manage conversations, and receive answers based on retrieved document chunks.

The implementation uses an owner-scoped vector index, so retrieval is limited to documents belonging to the authenticated user. PDF uploads retain one-based page numbers in chunk metadata, which supports source references in retrieved results and generated responses.

## Problem Statement

Organizations and individuals frequently store important information across documents, reports, manuals, and notes. Searching these files manually is slow, and general-purpose chat models can produce answers that are not grounded in the user's source material.

DocuMind AI addresses this problem by:

- Indexing each user's uploaded documents
- Restricting retrieval to that user's documents
- Returning relevant source chunks with document and chunk metadata
- Building an answer prompt from retrieved context
- Persisting chat conversations and messages

## Key Features

- Email/password registration and login
- JWT bearer authentication and protected API routes
- PDF, DOCX, and TXT document uploads
- Server-side document filename and MIME-type validation
- 25 MB maximum upload size by default
- Deterministic text chunking with configurable chunk overlap
- PDF page-number preservation
- Local Sentence Transformers embeddings using `all-MiniLM-L6-v2`
- Persistent Chroma vector storage
- Owner-scoped semantic document search
- Document-grounded chat responses with source metadata
- Conversation creation, listing, search, history, and deletion
- Paginated document and conversation lists
- PostgreSQL-backed users, documents, conversations, and messages
- Responsive React document and chat workspace
- Docker Compose deployment with health and readiness checks
- Swagger UI and ReDoc API documentation

## Technology Stack

### Frontend

- React 19
- TypeScript
- Vite
- React Router
- React Markdown
- Vitest, Testing Library, and jsdom

### Backend

- Python 3.11
- FastAPI
- Pydantic and Pydantic Settings
- SQLAlchemy 2 and AsyncPG
- Alembic
- PyJWT
- PyPDF
- python-docx
- Sentence Transformers
- ChromaDB
- Ollama HTTP API
- Cloudflare Workers AI REST API

### Data and Infrastructure

- PostgreSQL 16
- Ollama
- Docker and Docker Compose
- Nginx
- Persistent Docker volumes

## System Architecture

```text
React/Vite Frontend
        |
        | HTTP + Bearer JWT
        v
Nginx --> FastAPI API
              |
              +--> PostgreSQL
              |    - Users
              |    - Documents
              |    - Conversations
              |    - Messages
              |
              +--> Document parser
              |    - PDF
              |    - DOCX
              |    - TXT
              |
              +--> Sentence Transformers
              |    - all-MiniLM-L6-v2
              |
              +--> Chroma persistent vector store
              |
              +--> Ollama
                   - Gemma 3 4B (local default)
              |
              +--> Cloudflare Workers AI
                   - @cf/meta/llama-3.3-70b-instruct-fp8-fast (cloud option)
```

The Docker deployment runs PostgreSQL, Ollama, the FastAPI backend, and the Nginx frontend in a shared Compose network. Uploaded files and the persistent vector index are stored in Docker volumes.

## How the RAG Pipeline Works

1. **Authentication**
   - The user registers or logs in.
   - FastAPI issues a JWT bearer token.
   - Protected document, search, chat, and conversation routes validate the current user.

2. **Document upload**
   - The API checks the file extension and MIME type.
   - The file is stored in the configured document storage directory.
   - Document metadata is saved in PostgreSQL.

3. **Parsing**
   - PDF files are parsed page by page.
   - DOCX files are extracted from paragraphs.
   - TXT files are read as UTF-8 text.

4. **Chunking**
   - Text is split into character chunks of approximately 1,000 characters.
   - Adjacent chunks overlap by approximately 100 characters.
   - PDF page numbers are retained as chunk metadata.

5. **Embedding**
   - Each chunk is converted into a vector using `SentenceTransformer`.
   - The embedding model is configurable through `EMBEDDING_MODEL`.

6. **Vector indexing**
   - Chroma stores the chunk text, embedding, document ID, chunk index, owner ID, and available page metadata.
   - Vector IDs use the document ID and chunk index.

7. **Owner-scoped retrieval**
   - The user's question is converted into an embedding.
   - Chroma queries the collection with the authenticated owner's ID.
   - The API returns the configured number of most relevant chunks.

8. **Answer generation**
   - Retrieved text and source metadata are added to a constrained prompt.
   - The prompt instructs the model to answer only from the retrieved context.
   - The response is requested from the configured Ollama model.

9. **Conversation persistence**
   - Chat messages are associated with a conversation owned by the user.
   - Conversations and their message history can be listed and reopened.

## Installation and Setup

### Prerequisites

- Windows 10 or 11
- Docker Desktop with Docker Compose
- Git
- For native frontend/backend development:
  - Node.js 22
  - Python 3.11
  - Ollama
  - PostgreSQL 16

### 1. Clone the project

Open PowerShell and run:

```powershell
git clone https://github.com/Psahasra1124/Inventory-Management-System.git
cd Inventory-Management-System
```

> If this README is already in the downloaded project folder, continue from that folder instead of cloning again.

### 2. Configure environment variables

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Set strong values for:

```dotenv
POSTGRES_PASSWORD=<strong-random-password>
JWT_SECRET_KEY=<random-secret-of-at-least-32-characters>
```

Keep `.env` private and do not publish real secrets. The default Ollama model is `gemma3:4b`, the default embedding model is `all-MiniLM-L6-v2`, and the default upload limit is 25 MB.

#### Choose the LLM provider

Local Ollama development:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT_SECONDS=300
```

Cloudflare Workers AI deployment:

```dotenv
LLM_PROVIDER=cloudflare
CLOUDFLARE_ACCOUNT_ID=<your-cloudflare-account-id>
CLOUDFLARE_API_TOKEN=<your-cloudflare-api-token>
CLOUDFLARE_MODEL=@cf/meta/llama-3.3-70b-instruct-fp8-fast
CLOUDFLARE_TIMEOUT_SECONDS=90
```

For Render, set `LLM_PROVIDER=cloudflare`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, and `CLOUDFLARE_MODEL` in the backend service environment. `CLOUDFLARE_TIMEOUT_SECONDS` is optional and defaults to `90`. Keep the account ID and API token in Render only; neither value is sent to or exposed by the React/Vite frontend. The default model ID is listed in the [Cloudflare Workers AI model catalog](https://developers.cloudflare.com/workers-ai/models/llama-3.3-70b-instruct-fp8-fast/).

### 3. Docker setup (recommended)

Start Docker Desktop, then run:

```powershell
docker compose up --build -d
```

The backend container applies Alembic migrations before starting Uvicorn. Wait for the Ollama model initialization service to finish its first model download.

Open:

- Application: http://localhost:8080
- API documentation: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

Windows helper scripts are also included:

```powershell
.\START_PROJECT.bat
.\OPEN_APP.bat
.\CHECK_STATUS.bat
.\STOP_PROJECT.bat
```

### 4. Native setup

#### Start PostgreSQL

Create a database and user that match the `DATABASE_URL` in `.env`. The example configuration uses:

```text
Database: ai_document_rag
Port: 5433
User: postgres
```

The application converts PostgreSQL URLs to `postgresql+asyncpg://` for FastAPI. Alembic migrations use a synchronous `postgresql+psycopg2://` URL derived from the same `DATABASE_URL`; set `SYNC_DATABASE_URL` only if the deployment needs a separate migration connection. For hosted PostgreSQL providers such as Neon, use the provider's PostgreSQL URL as `DATABASE_URL` and keep credentials in the deployment's secret environment settings.

#### Start Ollama

Install Ollama for Windows, start the service, and pull the configured model:

```powershell
ollama pull gemma3:4b
ollama serve
```

Keep in mind that the local `.env.example` uses `http://ollama:11434` for Compose. Native backend execution should use `http://localhost:11434`.

#### Backend

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Frontend

Open a second PowerShell window:

```powershell
cd frontend
npm ci
npm run dev
```

The Vite development server normally runs at http://localhost:5173.

## Docker Deployment

The Compose deployment includes:

| Service | Container | Host port |
|---|---|---:|
| PostgreSQL | `rag_postgres` | `5433` |
| Ollama | `rag_ollama` | Internal only |
| Ollama model setup | `rag_ollama_model_init` | One-time task |
| FastAPI | `rag_backend` | `8000` |
| React/Nginx | `rag_frontend` | `8080` |

Useful commands:

```powershell
# Build and start
docker compose up --build -d

# View status
docker compose ps

# Follow application logs
docker compose logs -f backend frontend

# Check service state and logs
.\CHECK_STATUS.bat

# Stop containers without deleting persistent data
docker compose down

# Stop containers and remove volumes
docker compose down -v
```

Do not use `docker compose down -v` unless you intentionally want to delete PostgreSQL data, uploaded files, and the Chroma vector index.

## API Documentation

The API exposes Swagger UI at `/docs` and ReDoc at `/redoc`.

### Operations

| Method | Endpoint | Authentication | Description |
|---|---|---|---|
| `GET` | `/health` | No | Process liveness |
| `GET` | `/ready` | No | Database readiness |

### Authentication

| Method | Endpoint | Authentication | Description |
|---|---|---|---|
| `POST` | `/auth/register` | No | Register a user |
| `POST` | `/auth/login` | No | Obtain a JWT bearer token |
| `GET` | `/auth/me` | Bearer | Get the current user |

Example registration:

```powershell
$body = @{
    email = "user@example.com"
    password = "StrongPassword123"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/auth/register" `
  -ContentType "application/json" `
  -Body $body
```

Example login:

```powershell
$body = @{
    email = "user@example.com"
    password = "StrongPassword123"
} | ConvertTo-Json

$response = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/auth/login" `
  -ContentType "application/json" `
  -Body $body

$token = $response.access_token
$headers = @{ Authorization = "Bearer $token" }
```

### Documents

| Method | Endpoint | Authentication | Description |
|---|---|---|---|
| `GET` | `/documents` | Bearer | List/search documents with pagination |
| `POST` | `/documents/upload` | Bearer | Upload and index a PDF, DOCX, or TXT file |
| `DELETE` | `/documents/{document_id}` | Bearer | Delete an owned document and its vectors |
| `POST` | `/documents/search` | Bearer | Search document chunks semantically |

Example document search:

```powershell
$body = @{
    query = "What is the main conclusion?"
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/documents/search" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

### Chat

| Method | Endpoint | Authentication | Description |
|---|---|---|---|
| `POST` | `/chat` | Bearer | Generate a document-grounded answer |

Request example:

```json
{
  "query": "Summarize the key findings",
  "top_k": 5,
  "conversation_id": null
}
```

`conversation_id` is optional. When omitted, the API creates a new conversation and uses the first query to derive its title.

### Conversations

| Method | Endpoint | Authentication | Description |
|---|---|---|---|
| `GET` | `/conversations` | Bearer | List/search conversations |
| `GET` | `/conversations/{conversation_id}/messages` | Bearer | Load conversation message history |
| `DELETE` | `/conversations/{conversation_id}` | Bearer | Delete an owned conversation |

## Screenshots

The repository does not currently contain a screenshot asset directory. Add application captures to `docs/screenshots/` and embed them here to showcase the finished interface.

Suggested screenshots:

- Registration and login
- Document workspace
- Document upload and search
- Chat response with source metadata
- Conversation history

Example layout after screenshots are added:

```markdown
![Login](docs/screenshots/login.png)
![Document Workspace](docs/screenshots/dashboard.png)
![Document Chat](docs/screenshots/chat.png)
```

## Testing

The backend includes focused tests for authentication, uploads, parsing, chunking, indexing, retrieval, chat, persistence, migrations, CORS, and evaluation behavior. The frontend includes component and API tests.

Backend:

```powershell
cd backend
.venv\Scripts\Activate.ps1
python -m unittest discover -s tests
```

Frontend:

```powershell
cd frontend
npm test
npm run typecheck
npm run lint
npm run build
```

## Future Enhancements

The following items are potential roadmap ideas and are not confirmed as implemented:

- OCR and text extraction for scanned PDFs
- Additional formats such as PPTX and XLSX
- Streaming chat responses
- Reranking for improved retrieval quality
- Document-level response citation links in the UI
- Cloud-hosted embedding and vector providers
- Role-based access control
- Conversation sharing and export
- Automated retrieval and grounded-answer evaluation in the UI

## Author

**Psahasra1124**

- GitHub: https://github.com/Psahasra1124
- Project repository: https://github.com/Psahasra1124/Inventory-Management-System
