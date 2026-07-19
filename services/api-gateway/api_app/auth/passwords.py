# services/api-gateway/api_app/auth/passwords.py
"""Password hashing and verification.

Uses bcrypt directly (no passlib). bcrypt has a hard 72-byte input limit and
silently truncates longer inputs, so over-length passwords are rejected rather
than allowed to collide.
"""

from __future__ import annotations

import bcrypt

# bcrypt has a hard 72-byte input limit; longer inputs are silently truncated,
# so we reject them rather than let two different passwords collide.
_MAX_BYTES = 72


def hash_password(plain: str) -> str:
    """Hash a plaintext password. Returns a utf-8 string safe to store."""
    raw = plain.encode("utf-8")
    if len(raw) > _MAX_BYTES:
        raise ValueError("Password too long (max 72 bytes).")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plaintext password against a stored bcrypt hash."""
    raw = plain.encode("utf-8")
    if len(raw) > _MAX_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # malformed / non-bcrypt hash (e.g. a google-only user with NULL hash)
        return False
