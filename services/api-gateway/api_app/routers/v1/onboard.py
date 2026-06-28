from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ops_common.config import settings
from ops_common.db import get_db
from ops_common.logging import get_logger
from app.pipeline import start_onboarding, complete_onboarding
from api_app.spark_trigger import trigger_analytics_async


logger = get_logger(__name__)

router = APIRouter()

_ALLOWED_SUFFIXES = {".csv"}


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Only CSV is supported.",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    unique = f"{uuid.uuid4().hex}{suffix}"
    dest = settings.upload_dir / unique

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    file.file.close()

    logger.info("Saved upload", extra={"original": file.filename, "stored": unique})
    return dest


def _stored_path_for_dataset(dataset_id: int, session: Session) -> Path:
    from ops_common.domain.models import Dataset

    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    matches = sorted(settings.upload_dir.glob(f"*{Path(dataset.source_filename).suffix}"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Stored upload file for this dataset was not found.",
        )
    return matches[-1]


# ============================================================
# Request / response models
# ============================================================

class ConfirmColumnIn(BaseModel):
    column_name: str
    domain: str | None = None
    metric_name: str | None = None
    role: str = Field(default="skip")


class ConfirmRequest(BaseModel):
    dataset_id: int
    stored_path: str
    columns: list[ConfirmColumnIn]


class StartResponse(BaseModel):
    dataset_id: int
    business_name: str
    industry: str | None
    row_count: int
    stored_path: str
    suggestions: list[dict[str, Any]]


class CompleteResponse(BaseModel):
    dataset_id: int
    config_version: int
    hub_rows_written: int
    features_collected: int
    features_skipped: int
    validation: dict[str, Any]


# ============================================================
# Endpoints
# ============================================================

@router.post("/onboard/start", response_model=StartResponse)
def onboard_start(
    business_name: str = Form(...),
    industry: str | None = Form(default=None),
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
) -> StartResponse:
    stored = _save_upload(file)

    try:
        result = start_onboarding(
            session=session,
            csv_path=stored,
            business_name=business_name,
            industry=industry,
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("onboard/start failed")
        raise

    return StartResponse(
        dataset_id=result.dataset_id,
        business_name=result.business_name,
        industry=result.industry,
        row_count=result.row_count,
        stored_path=str(stored),
        suggestions=result.suggestions,
    )


@router.post("/onboard/confirm", response_model=CompleteResponse)
def onboard_confirm(
    payload: ConfirmRequest,
    session: Session = Depends(get_db),
) -> CompleteResponse:
    stored = Path(payload.stored_path)
    if not stored.exists():
        raise HTTPException(
            status_code=404,
            detail="Stored upload no longer available; please re-upload.",
        )

    confirmed = [c.model_dump() for c in payload.columns]

    try:
        result = complete_onboarding(
            session=session,
            dataset_id=payload.dataset_id,
            csv_path=stored,
            confirmed=confirmed,
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("onboard/confirm failed")
        raise

    trigger_analytics_async(result.dataset_id)

    return CompleteResponse(
        dataset_id=result.dataset_id,
        config_version=result.config_version,
        hub_rows_written=result.hub_rows_written,
        features_collected=result.features_collected,
        features_skipped=result.features_skipped,
        validation=result.validation,
    )