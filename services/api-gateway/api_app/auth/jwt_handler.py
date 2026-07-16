# services/api-gateway/api_app/auth/jwt_handler.py
# Token crypto only — no DB. Access = JWT (stateless). Refresh = opaque token.

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from ops_common.config import settings


# ---- Access token (JWT, short-lived, verified without DB) ----

def create_access_token(user_id: int, email: str, roles: list[str],
                        permissions: list[str]) -> str:
    """Sign a short-lived access JWT carrying identity + authorization claims."""
    now = datetime.now(timezone.utc)
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
    """
    Mint a new refresh token.
    Returns (raw_token, token_hash, expires_at):
      raw_token   -> sent to the client, never stored.
      token_hash  -> stored in auth.refresh_tokens.token_hash.
      expires_at  -> stored in auth.refresh_tokens.expires_at.
    """
    raw = secrets.token_urlsafe(48)
    token_hash = hash_refresh_token(raw)
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    return raw, token_hash, expires_at


def hash_refresh_token(raw: str) -> str:
    """sha256 of the raw refresh token — used both to store and to look it up."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()