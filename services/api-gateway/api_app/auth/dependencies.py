# services/api-gateway/api_app/auth/dependencies.py
# FastAPI auth guards: extract token, verify user, enforce roles/permissions.

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from sqlalchemy.orm import Session

from ops_common.db import get_db
from ops_common.logging import get_logger

from api_app.auth.jwt_handler import decode_access_token

logger = get_logger(__name__)

# auto_error=False so we can raise our own 401 with a clean message.
_bearer = HTTPBearer(auto_error=False)


class CurrentUser:
    """The authenticated caller, resolved from a verified access token + DB check."""

    def __init__(self, user_id: int, email: str,
                 roles: list[str], permissions: list[str]) -> None:
        self.user_id = user_id
        self.email = email
        self.roles = roles
        self.permissions = permissions

    def has_permission(self, code: str) -> bool:
        return code in self.permissions

    def has_role(self, name: str) -> bool:
        return name in self.roles


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_db),
) -> CurrentUser:
    """
    Verify the bearer token and confirm the user still exists and is active.
    Token carries roles/permissions (fast path), but we re-check is_active in
    the DB so a disabled account can't keep using an unexpired token.
    """
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(payload["sub"])

    # Re-check the account is still active (token can't outlive a disable).
    try:
        row = session.execute(
            text("SELECT is_active FROM auth.users WHERE id = :id"),
            {"id": user_id},
        ).fetchone()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read auth.users during auth")
        raise HTTPException(status_code=503, detail="Auth layer unavailable.")

    if row is None or not row[0]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account inactive or not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return CurrentUser(
        user_id=user_id,
        email=payload.get("email", ""),
        roles=payload.get("roles", []),
        permissions=payload.get("permissions", []),
    )


def require_permission(code: str):
    """Dependency factory: 403 unless the caller holds `code`."""
    def _guard(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not user.has_permission(code):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {code}",
            )
        return user
    return _guard


def require_role(name: str):
    """Dependency factory: 403 unless the caller holds role `name`."""
    def _guard(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not user.has_role(name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {name}",
            )
        return user
    return _guard