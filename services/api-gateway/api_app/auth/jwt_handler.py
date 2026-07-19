# services/api-gateway/api_app/auth/jwt_handler.py
"""Token cryptography for authentication — no database access.

Access tokens are stateless, signed JWTs carrying identity and authorization
claims, verifiable without a database round-trip. Refresh tokens are opaque
random strings; only their sha256 hash is stored, so a database leak cannot
reveal a usable token.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from ops_common.config import settings

# ---- Access token (JWT, short-lived, verified without DB) ----


def create_access_token(
    user_id: int, email: str, roles: list[str], permissions: list[str]
) -> str:
    """Sign a short-lived access JWT carrying identity + authorization claims."""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "email": email,
        "roles": roles,
        "permissions": permissions,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """Decode and verify an access JWT. Raises jwt.PyJWTError on invalid/expired."""
    payload = jwt.decode(
        token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Not an access token.")
    return payload


# ---- Refresh token (opaque random string, sha256-hashed for DB storage) ----


def create_refresh_token() -> tuple[str, str, datetime]:
    """Mint a new refresh token.

    Returns:
        A ``(raw_token, token_hash, expires_at)`` tuple where ``raw_token`` is
        sent to the client and never stored, ``token_hash`` is persisted in
        ``auth.refresh_tokens.token_hash``, and ``expires_at`` is persisted in
        ``auth.refresh_tokens.expires_at``.
    """
    raw = secrets.token_urlsafe(48)
    token_hash = hash_refresh_token(raw)
    expires_at = datetime.now(UTC) + timedelta(
        days=settings.refresh_token_expire_days
    )
    return raw, token_hash, expires_at


def hash_refresh_token(raw: str) -> str:
    """sha256 of the raw refresh token — used both to store and to look it up."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
