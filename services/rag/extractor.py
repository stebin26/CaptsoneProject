# Text extraction — PDF / DOCX / TXT to plain text with page metadata.
# First stage of the RAG ingest pipeline; feeds the chunker.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedPage:
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str
    pages: list[ExtractedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext in ("pdf", "docx", "txt", "md"):
        return "txt" if ext == "md" else ext
    return ext or "unknown"


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> list[ExtractedPage]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[ExtractedPage] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — a single bad page shouldn't kill the doc
            text = ""
        pages.append(ExtractedPage(page_number=i + 1, text=_clean(text)))
    return pages


def _extract_docx(path: Path) -> list[ExtractedPage]:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    parts: list[str] = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    # docx has no intrinsic page breaks accessible here; treat as a single page.
    text = _clean("\n".join(parts))
    return [ExtractedPage(page_number=1, text=text)] if text else []


def _extract_txt(path: Path) -> list[ExtractedPage]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _clean(raw)
    return [ExtractedPage(page_number=1, text=text)] if text else []


# ---------------------------------------------------------------------------
# Cleaning + entry point
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    # Normalize whitespace, drop null bytes, collapse excessive blank lines.
    text = text.replace("\x00", " ")
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip():
            cleaned.append(ln)
            blank = 0
        else:
            blank += 1
            if blank <= 1:
                cleaned.append("")
    return "\n".join(cleaned).strip()


def extract_document(path: str | Path, filename: str | None = None) -> ExtractedDocument:
    """Extract text from a supported file. Raises ValueError on unsupported type."""
    p = Path(path)
    name = filename or p.name
    ftype = detect_file_type(name)

    if ftype == "pdf":
        pages = _extract_pdf(p)
    elif ftype == "docx":
        pages = _extract_docx(p)
    elif ftype == "txt":
        pages = _extract_txt(p)
    else:
        raise ValueError(f"Unsupported file type: {ftype} ({name})")

    return ExtractedDocument(filename=name, file_type=ftype, pages=pages)