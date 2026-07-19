# services/api-gateway/api_app/auth/dependencies.py
"""FastAPI authentication and authorization guards.

Provides the dependencies routes use to protect themselves: extracting and
verifying the bearer access token, re-confirming the account is still active in
the database, and enforcing role- and permission-based access control. The
access token carries roles and permissions for a fast path, but ``is_active`` is
re-checked against the database so a disabled account cannot keep using an
unexpired token.
"""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ops_common.db import get_db
from ops_common.logging import get_logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from api_app.auth.jwt_handler import decode_access_token

logger = get_logger(__name__)

# auto_error=False so we can raise our own 401 with a clean message.
_bearer = HTTPBearer(auto_error=False)


class CurrentUser:
    """The authenticated caller, resolved from a verified access token + DB check."""

    def __init__(
        self, user_id: int, email: str, roles: list[str], permissions: list[str]
    ) -> None:
        """Store the caller's resolved identity and access.

        Args:
            user_id: Authenticated user's id.
            email: Authenticated user's email.
            roles: Role names granted to the user.
            permissions: Permission codes granted to the user.
        """
        self.user_id = user_id
        self.email = email
        self.roles = roles
        self.permissions = permissions

    def has_permission(self, code: str) -> bool:
        """Return whether the caller holds the given permission code."""
        return code in self.permissions

    def has_role(self, name: str) -> bool:
        """Return whether the caller holds the given role name."""
        return name in self.roles


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_db),
) -> CurrentUser:
    """Verify the bearer token and confirm the user is still active.

    The token carries roles and permissions (the fast path), but ``is_active``
    is re-read from the database so a disabled account cannot keep using an
    unexpired token.

    Args:
        creds: Bearer credentials extracted from the Authorization header.
        session: Active database session.

    Returns:
        The resolved ``CurrentUser`` for the request.

    Raises:
        HTTPException: 401 if the token is missing, expired, invalid, or the
            account is gone; 503 if the auth store cannot be read.
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
        ) from None
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        # The token verified but its subject claim is unusable, so it cannot
        # identify anyone — treated as an invalid token, not a server error.
        logger.warning(
            "Access token carried an unusable subject claim",
            extra={"subject": payload.get("sub")},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    # Re-check the account is still active (token can't outlive a disable).
    try:
        row = session.execute(
            text("SELECT is_active FROM auth.users WHERE id = :id"),
            {"id": user_id},
        ).fetchone()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read auth.users during auth")
        raise HTTPException(status_code=503, detail="Auth layer unavailable.") from None

    if row is None or not row[0]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

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
