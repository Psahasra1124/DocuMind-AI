DocuMind AI - PDF page citation patch

Replace the four files at their matching paths:
- backend/app/services/document_parsers/parsers.py
- backend/app/services/document_chunking.py
- backend/app/services/document_indexing.py
- backend/app/services/rag.py

No change is required in retrieval.py or vector_store/chroma.py:
retrieval already preserves the metadata dictionary, and Chroma stores it.

Before applying, back up your project.

After replacing the files, rebuild the backend image (a restart alone does not
copy changed source files into an existing image):

    docker compose up -d --build backend

Then check:

    docker compose ps
    docker compose logs --tail=100 backend

Re-upload the PDFs you want to cite. Existing indexed chunks do not gain page
metadata automatically.

PDF citations use actual 1-based PDF page positions. TXT/DOCX keep their
existing extraction and chunking behavior and show chunk references rather
than invented page numbers.

This patch was prepared against the source files shared in this conversation.
Run the backend test suite before relying on it for deployment.
