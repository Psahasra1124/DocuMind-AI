from collections.abc import Iterable, Iterator


def iter_text_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> Iterator[str]:
    """Yield deterministic, overlapping character chunks without retaining all."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return

    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        if end == len(text):
            break
        start += step


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into deterministic, overlapping character chunks."""
    return list(iter_text_chunks(text, chunk_size, chunk_overlap))


def iter_page_chunks(
    pages: Iterable[tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> Iterator[tuple[int, str]]:
    """Yield page chunks incrementally while retaining their page numbers."""
    for page_number, page_text in pages:
        for chunk in iter_text_chunks(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            yield page_number, chunk


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[tuple[int, str]]:
    """Chunk each PDF page independently and retain its 1-based page number."""
    return list(iter_page_chunks(pages, chunk_size, chunk_overlap))
