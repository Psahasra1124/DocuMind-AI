# DocuMind AI — Portfolio & Interview Guide

## One-line summary

DocuMind AI is a multi-user document intelligence platform that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded PDF, DOCX, and TXT files, while returning source evidence so users can inspect where an answer came from.

## 60–90 second project pitch

“I built DocuMind AI, a full-stack document intelligence platform using React and TypeScript, FastAPI, PostgreSQL, ChromaDB, Sentence Transformers, and Ollama. Users register, upload documents, and ask natural-language questions about their own files. The backend extracts text, chunks it, creates embeddings, stores them in a vector database, retrieves relevant chunks for each question, and passes that context to a local language model. Responses include source evidence so the user can trace answers back to the retrieved document content.

A major engineering focus was tenant isolation: document ownership is carried through indexing and enforced at the vector retrieval boundary, rather than relying only on frontend filtering. I also implemented persistent chat history, migrations, document lifecycle handling, API error handling, upload limits, and retrieval/grounding evaluation datasets. The project is containerized for local demonstration with Docker Compose.”

## Recommended live demo flow

1. Open the app and create a user account.
2. Upload a small, text-based PDF or DOCX containing a few facts and section headings.
3. Wait for indexing to complete.
4. Ask a specific question whose answer appears in the uploaded file.
5. Expand the displayed source evidence and show the source text/document reference.
6. Ask a question that is not covered by the file and discuss how retrieval evidence affects the response.
7. Show conversation history and reload the page to demonstrate persistence.
8. Explain that a second account has a separate ownership scope; do not use another user's private file as a demo fixture.
9. Open `/docs` to show the FastAPI API contract.
10. Briefly explain the retrieval evaluation dataset and its limitations.

Use a small, clean, non-sensitive sample document. Do not upload confidential company, customer, or personal records into a portfolio demo.

## Architecture walkthrough

```text
React + TypeScript
      │ authenticated HTTP
      ▼
FastAPI API ───────── JWT authentication
      │
      ├── PostgreSQL: users, documents, conversations, messages
      ├── File storage: original uploaded documents
      └── Indexing pipeline
              │
              ├── PDF / DOCX / TXT parsing
              ├── deterministic chunking
              ├── Sentence Transformers embeddings
              └── ChromaDB vector persistence

Question → owner-scoped retrieval → context assembly → Ollama LLM
                                      │
                                      └── answer + source evidence
```

## Engineering decisions worth discussing

- **Why RAG instead of fine-tuning?** The application needs to answer from changing user documents. Retrieval lets the system ground each request in the relevant current content without retraining model weights.
- **Why a vector database?** Embeddings support semantic matching when the user's wording differs from the document wording.
- **Why PostgreSQL and ChromaDB?** PostgreSQL stores relational application state and ownership; ChromaDB stores embedding vectors and retrieval metadata.
- **Why owner-scoped retrieval?** Authorization must hold at the data access boundary. Hiding another user's document in the UI is not sufficient.
- **Why source evidence?** It makes responses more inspectable and supports debugging and user trust. A citation is evidence of retrieved context, not a guarantee that every generated statement is correct.
- **Why local Ollama?** It enables a reproducible local demo without sending document content to a paid hosted LLM API. The trade-off is local hardware and inference latency.
- **Why migrations?** Schema changes are versioned and repeatable instead of depending on manual database edits.

## Evaluation — describe the results accurately

The repository includes controlled retrieval and deterministic grounding evaluation datasets. These are useful regression checks for the included synthetic corpus. They are not independent industry benchmarks and do not establish production answer accuracy, factuality, hallucination rate, or generalization to arbitrary customer documents. Present the metrics together with this limitation.

## Security and reliability controls implemented

- JWT-based authentication.
- User ownership checks for documents and conversations.
- Owner-scoped vector retrieval; no unrestricted authenticated fallback.
- Upload type validation and configurable upload size limit.
- Persistent relational and vector storage.
- Document deletion lifecycle handling.
- Alembic schema migrations.
- Generic internal error responses with server-side exception logging.
- API liveness and database-readiness endpoints.

## Current limitations to acknowledge honestly

- Local/demo deployment; no hardened cloud deployment or managed secrets service is included.
- Ollama inference is local and can be slow on CPU-only hardware.
- First startup downloads the Ollama model and embedding model.
- OCR/scanned-PDF extraction is not part of the verified feature set.
- Indexing is synchronous from the user's perspective; a production scale-out would normally move long-running ingestion into a background job queue with status tracking and retries.
- Production observability, rate limiting, formal threat modeling, backup automation, and load testing remain future work.

## Resume-ready project description

**DocuMind AI – RAG-Based Document Intelligence Platform**  
Built a full-stack document intelligence platform using React, TypeScript, FastAPI, PostgreSQL, ChromaDB, Sentence Transformers, and Ollama. Implemented document parsing, chunking, semantic retrieval, source-grounded chat, JWT authentication, owner-scoped retrieval, persistent conversations, and retrieval/grounding evaluation. Containerized the application with Docker Compose for repeatable local deployment.

## Resume skills demonstrated

Python, FastAPI, React, TypeScript, REST APIs, PostgreSQL, SQLAlchemy, JWT Authentication, Retrieval-Augmented Generation (RAG), Embeddings, Sentence Transformers, ChromaDB, Ollama, Docker, Docker Compose, Alembic, Pytest, Vitest.
