from pathlib import Path

from app.services.document_parsers.base import DocumentParser


class PDFDocumentParser(DocumentParser):
    def parse_pages(self, file_path: Path) -> list[tuple[int, str]]:
        """Extract text page by page, retaining 1-based PDF page numbers."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    def parse(self, file_path: Path) -> str:
        """Keep the original parser interface for existing callers."""
        return "\n".join(text for _, text in self.parse_pages(file_path))


class DOCXDocumentParser(DocumentParser):
    def parse(self, file_path: Path) -> str:
        from docx import Document

        document = Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)


class TXTDocumentParser(DocumentParser):
    def parse(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")
