"""Dataset onboarding API endpoints.

Handles CSV upload and the two-stage onboarding flow: start (profile columns and
suggest a mapping) and confirm (persist the confirmed mapping, load the hub, and
trigger analytics).
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from app.pipeline import complete_onboarding, start_onboarding
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from ops_common.config import settings
from ops_common.db import get_db
from ops_common.logging import get_logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api_app.auth.dependencies import require_permission
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

    matches = sorted(
        settings.upload_dir.glob(f"*{Path(dataset.source_filename).suffix}")
    )
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
    """One column's confirmed mapping decision."""
    column_name: str
    domain: str | None = None
    metric_name: str | None = None
    role: str = Field(default="skip")


class ConfirmRequest(BaseModel):
    """Request body confirming a dataset's column mapping."""
    dataset_id: int
    stored_path: str
    columns: list[ConfirmColumnIn]


class StartResponse(BaseModel):
    """Response from onboarding start: profile plus mapping suggestions."""
    dataset_id: int
    business_name: str
    industry: str | None
    row_count: int
    stored_path: str
    suggestions: list[dict[str, Any]]


class CompleteResponse(BaseModel):
    """Response from onboarding confirm: load and feature counts."""
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
    _user=Depends(require_permission("dataset:upload")),
) -> StartResponse:
    """Start onboarding: save the upload, profile columns, suggest a mapping.

    Args:
        business_name: Name of the business the dataset belongs to.
        industry: Optional industry label.
        file: The uploaded CSV file.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``dataset:upload``.

    Returns:
        The new dataset's profile and suggested column mapping.
    """
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
    _user=Depends(require_permission("mapping:confirm")),
) -> CompleteResponse:
    """Complete onboarding: persist the mapping, load the hub, trigger analytics.

    Args:
        payload: The dataset id, stored file path, and confirmed column decisions.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``mapping:confirm``.

    Returns:
        Hub-load and feature counts plus the validation report.

    Raises:
        HTTPException: 404 if the stored upload is no longer available.
    """
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
