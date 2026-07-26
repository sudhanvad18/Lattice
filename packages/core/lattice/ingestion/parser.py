"""
Document parsers for various file formats.

Each parser takes a file path or raw content and returns a Document
with clean, normalized text content ready for chunking.

Design choice: parsers are stateless functions, not classes. This keeps
them simple, testable, and composable. The router pattern at the bottom
selects the right parser based on file extension.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from lattice.ingestion.models import Document, DocumentMetadata, DocumentType

logger = structlog.get_logger()


def parse_pdf(path: Path) -> Document:
    """Extract text from a PDF file.

    Uses pypdf which handles most standard PDFs without external deps.
    For scanned PDFs (images), you'd need OCR — out of scope for now.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    content = "\n\n".join(pages)

    metadata = DocumentMetadata(
        source=str(path),
        title=reader.metadata.title if reader.metadata else path.stem,
        author=reader.metadata.author if reader.metadata else None,
        doc_type=DocumentType.PDF,
    )

    logger.info("pdf_parsed", path=str(path), pages=len(reader.pages), chars=len(content))
    return Document(content=content, metadata=metadata)


def parse_markdown(path: Path) -> Document:
    """Parse a markdown file, preserving structure for chunking."""
    content = path.read_text(encoding="utf-8")

    metadata = DocumentMetadata(
        source=str(path),
        title=_extract_md_title(content) or path.stem,
        doc_type=DocumentType.MARKDOWN,
    )

    logger.info("markdown_parsed", path=str(path), chars=len(content))
    return Document(content=content, metadata=metadata)


def parse_html(path: Path) -> Document:
    """Parse HTML to clean text, stripping tags but preserving structure."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    raw_html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    title = soup.title.string if soup.title else path.stem

    # Convert HTML → markdown (preserves headings, lists, links)
    content = markdownify(raw_html, heading_style="ATX", strip=["script", "style"])
    content = _normalize_whitespace(content)

    metadata = DocumentMetadata(
        source=str(path),
        title=title,
        doc_type=DocumentType.HTML,
    )

    logger.info("html_parsed", path=str(path), chars=len(content))
    return Document(content=content, metadata=metadata)


def parse_text(path: Path) -> Document:
    """Parse a plain text file."""
    content = path.read_text(encoding="utf-8")

    metadata = DocumentMetadata(
        source=str(path),
        title=path.stem,
        doc_type=DocumentType.TEXT,
    )

    return Document(content=content, metadata=metadata)


def parse_content(content: str, source: str = "inline", doc_type: DocumentType = DocumentType.TEXT) -> Document:
    """Parse raw text content directly (not from a file)."""
    metadata = DocumentMetadata(
        source=source,
        doc_type=doc_type,
    )
    return Document(content=content, metadata=metadata)


# --- Router ---

PARSER_MAP: dict[str, callable] = {
    ".pdf": parse_pdf,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_text,
}


def parse_file(path: Path) -> Document:
    """Route a file to the appropriate parser based on extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    parser = PARSER_MAP.get(ext)
    if parser is None:
        logger.warning("unsupported_format", path=str(path), extension=ext)
        return parse_text(path)

    return parser(path)


# --- Helpers ---

def _extract_md_title(content: str) -> str | None:
    """Extract the first H1 heading from markdown content."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph structure."""
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
