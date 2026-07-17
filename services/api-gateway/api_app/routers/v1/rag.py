from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ops_common.config import settings
from ops_common.db import get_db
from ops_common.logging import get_logger
from api_app.auth.dependencies import require_permission

logger = get_logger(__name__)

router = APIRouter()

# The RAG package lives in services/rag; make it importable inside the API image.
_RAG_PATH = os.getenv("OPS_RAG_PATH", "/app/services/rag")
if _RAG_PATH not in sys.path:
    sys.path.insert(0, _RAG_PATH)

from pipeline import index_document, register_document  # noqa: E402
from qa_chain import answer_question                     # noqa: E402
from vector_store import delete_document, list_documents  # noqa: E402


_ALLOWED_EXT = {"pdf", "docx", "txt", "md"}


# ============================================================
# Response models
# ============================================================

class UploadedDocOut(BaseModel):
    document_id: int
    filename: str
    status: str


class UploadResponseOut(BaseModel):
    dataset_id: int
    accepted: list[UploadedDocOut]
    rejected: list[dict[str, Any]]


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str | None
    status: str
    chunk_count: int
    error_detail: str | None
    uploaded_at: str
    indexed_at: str | None


class SourceOut(BaseModel):
    filename: str
    page_number: int | None
    distance: float


class QueryResponseOut(BaseModel):
    dataset_id: int
    question: str
    answer: str
    grounded: bool
    llm_used: bool
    used_chunks: int
    sources: list[SourceOut]


# ============================================================
# Helpers
# ============================================================

def _ext(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def _save_upload(dataset_id: int, upload: UploadFile) -> tuple[Path, int]:
    settings.rag_upload_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{dataset_id}_{uuid.uuid4().hex}_{Path(upload.filename).name}"
    dest = settings.rag_upload_dir / safe
    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            size += len(chunk)
    return dest, size


def _index_task(dataset_id: int, document_id: int, path: str, filename: str) -> None:
    # Runs in the background so the upload request returns immediately.
    try:
        index_document(dataset_id, document_id, path, filename)
    except Exception:  # noqa: BLE001
        logger.exception("Background indexing crashed", extra={"document_id": document_id})


# ============================================================
# Endpoints
# ============================================================

@router.post("/rag/{dataset_id}/upload", response_model=UploadResponseOut)
def upload_documents(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    business_name: str | None = Form(default=None),
    _session: Session = Depends(get_db),
    _user=Depends(require_permission("documents:upload")),
) -> UploadResponseOut:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    accepted: list[UploadedDocOut] = []
    rejected: list[dict[str, Any]] = []

    for upload in files:
        fname = Path(upload.filename or "").name
        if not fname:
            rejected.append({"filename": upload.filename, "reason": "empty filename"})
            continue
        if _ext(fname) not in _ALLOWED_EXT:
            rejected.append({"filename": fname, "reason": f"unsupported type .{_ext(fname)}"})
            continue

        try:
            stored_path, size = _save_upload(dataset_id, upload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed saving upload")
            rejected.append({"filename": fname, "reason": f"save failed: {exc}"})
            continue

        # Register the doc row up front (pending) so the UI shows it immediately,
        # then index in the background.
        document_id = register_document(
            dataset_id=dataset_id,
            business_name=business_name,
            filename=fname,
            file_size=size,
        )
        background_tasks.add_task(
            _index_task, dataset_id, document_id, str(stored_path), fname
        )
        accepted.append(UploadedDocOut(document_id=document_id, filename=fname,
                                       status="pending"))

    return UploadResponseOut(dataset_id=dataset_id, accepted=accepted, rejected=rejected)


@router.get("/rag/{dataset_id}/documents", response_model=list[DocumentOut])
def get_documents(
    dataset_id: int,
    _session: Session = Depends(get_db),
    _user=Depends(require_permission("documents:read")),
) -> list[DocumentOut]:
    try:
        rows = list_documents(dataset_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to list RAG documents")
        raise HTTPException(status_code=503, detail="Document store unavailable.")

    return [
        DocumentOut(
            id=r["id"],
            filename=r["filename"],
            file_type=r.get("file_type"),
            status=r["status"],
            chunk_count=r.get("chunk_count") or 0,
            error_detail=r.get("error_detail"),
            uploaded_at=str(r["uploaded_at"]),
            indexed_at=str(r["indexed_at"]) if r.get("indexed_at") else None,
        )
        for r in rows
    ]


class QueryIn(BaseModel):
    question: str
    top_k: int | None = None


@router.post("/rag/{dataset_id}/query", response_model=QueryResponseOut)
def query_documents(
    dataset_id: int,
    body: QueryIn,
    _session: Session = Depends(get_db),
    _user=Depends(require_permission("documents:read")),
) -> QueryResponseOut:
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        result = answer_question(dataset_id, question, top_k=body.top_k)
    except Exception:  # noqa: BLE001
        logger.exception("RAG query failed")
        raise HTTPException(status_code=503, detail="RAG query failed.")

    return QueryResponseOut(
        dataset_id=dataset_id,
        question=question,
        answer=result.answer,
        grounded=result.grounded,
        llm_used=result.llm_used,
        used_chunks=result.used_chunks,
        sources=[SourceOut(**s) for s in result.sources],
    )


@router.delete("/rag/{dataset_id}/documents/{document_id}")
def remove_document(
    dataset_id: int,
    document_id: int,
    _session: Session = Depends(get_db),
    _user=Depends(require_permission("documents:upload")),
) -> dict[str, Any]:
    try:
        deleted = delete_document(dataset_id, document_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to delete document")
        raise HTTPException(status_code=503, detail="Delete failed.")
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True, "document_id": document_id}