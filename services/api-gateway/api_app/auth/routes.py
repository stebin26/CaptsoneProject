# services/api-gateway/api_app/auth/routes.py
"""Authentication and session endpoints for the API gateway.

Exposes the local email/password auth surface: admin-only user registration,
login, access-token refresh, single-session and all-session logout, and an
identity lookup for the current caller. Access tokens are short-lived JWTs;
refresh tokens are opaque, hashed, and stored in ``auth.refresh_tokens`` so they
can be revoked server-side. Role and permission resolution is read fresh from the
database at token-issue time, keeping RBAC the single source of truth.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from ops_common.db import get_db
from ops_common.logging import get_logger
from pydantic import BaseModel, EmailStr
from sqlalchemy import text
from sqlalchemy.orm import Session

from api_app.auth.dependencies import (
    CurrentUser,
    get_current_user,
    require_permission,
)
from api_app.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
)
from api_app.auth.passwords import hash_password, verify_password

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# Request / response models
# ============================================================


class RegisterIn(BaseModel):
    """Request body for admin-driven user registration."""

    email: EmailStr
    password: str
    full_name: str | None = None
    roles: list[str] = ["Viewer"]


class LoginIn(BaseModel):
    """Request body for email/password login."""

    email: EmailStr
    password: str


class RefreshIn(BaseModel):
    """Request body carrying a refresh token to exchange for a new access token."""

    refresh_token: str


class LogoutIn(BaseModel):
    """Request body carrying the refresh token to revoke on logout."""

    refresh_token: str


class TokenOut(BaseModel):
    """Response returning a new access token paired with its refresh token."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AccessOut(BaseModel):
    """Response returning a standalone access token (used by refresh)."""

    access_token: str
    token_type: str = "bearer"


class MeOut(BaseModel):
    """Response describing the authenticated caller's identity and access."""

    user_id: int
    email: str
    full_name: str | None
    roles: list[str]
    permissions: list[str]


# ============================================================
# Helpers
# ============================================================


def _fetch_roles(session: Session, user_id: int) -> list[str]:
    """Return the role names assigned to a user, ordered alphabetically.

    Args:
        session: Active database session.
        user_id: Id of the user whose roles are being read.

    Returns:
        The user's role names; empty if the user has no roles.
    """
    rows = session.execute(
        text(
            """
            SELECT r.name
            FROM auth.user_roles ur
            JOIN auth.roles r ON r.id = ur.role_id
            WHERE ur.user_id = :uid
            ORDER BY r.name
            """
        ),
        {"uid": user_id},
    ).fetchall()
    return [r[0] for r in rows]


def _fetch_permissions(session: Session, user_id: int) -> list[str]:
    """Return the distinct permission codes granted to a user via their roles.

    Args:
        session: Active database session.
        user_id: Id of the user whose permissions are being read.

    Returns:
        The distinct permission codes, ordered alphabetically; empty if none.
    """
    rows = session.execute(
        text(
            """
            SELECT DISTINCT p.code
            FROM auth.user_roles ur
            JOIN auth.role_permissions rp ON rp.role_id = ur.role_id
            JOIN auth.permissions p ON p.id = rp.permission_id
            WHERE ur.user_id = :uid
            ORDER BY p.code
            """
        ),
        {"uid": user_id},
    ).fetchall()
    return [r[0] for r in rows]


def _issue_tokens(
    session: Session, request: Request, user_id: int, email: str
) -> TokenOut:
    """Mint an access JWT and a DB-stored refresh token for a verified user.

    Resolves the user's roles and permissions, embeds them in a short-lived
    access token, persists a hashed refresh token (with the caller's user-agent
    and IP for auditing), stamps ``last_login``, and commits.

    Args:
        session: Active database session.
        request: Incoming request, used to capture user-agent and client IP.
        user_id: Id of the authenticated user.
        email: Email of the authenticated user, embedded in the access token.

    Returns:
        A ``TokenOut`` with the access token and the raw (unhashed) refresh token.
    """
    roles = _fetch_roles(session, user_id)
    permissions = _fetch_permissions(session, user_id)

    access = create_access_token(user_id, email, roles, permissions)
    raw_refresh, token_hash, expires_at = create_refresh_token()

    session.execute(
        text(
            """
            INSERT INTO auth.refresh_tokens
                (user_id, token_hash, user_agent, ip_address, expires_at)
            VALUES (:uid, :h, :ua, :ip, :exp)
            """
        ),
        {
            "uid": user_id,
            "h": token_hash,
            "ua": request.headers.get("user-agent"),
            "ip": request.client.host if request.client else None,
            "exp": expires_at,
        },
    )
    session.execute(
        text("UPDATE auth.users SET last_login = now() WHERE id = :uid"),
        {"uid": user_id},
    )
    session.commit()

    return TokenOut(access_token=access, refresh_token=raw_refresh)


# ============================================================
# Endpoints
# ============================================================


@router.post("/auth/register", response_model=MeOut, status_code=201)
def register(
    body: RegisterIn,
    session: Session = Depends(get_db),
    _admin: CurrentUser = Depends(require_permission("user:manage")),
) -> MeOut:
    """Create a local email/password user and assign roles (admin only).

    Guarded by the ``user:manage`` permission. Rejects duplicate emails and
    unknown role names before creating the user, then attaches the requested
    roles.

    Args:
        body: New user's email, password, optional name, and requested roles.
        session: Active database session.
        _admin: Authenticated admin caller, injected to enforce the permission.

    Returns:
        The created user's identity, roles, and resolved permissions.

    Raises:
        HTTPException: 409 if the email exists, 400 if a role is unknown or the
            password is rejected by the hashing policy.
    """
    exists = session.execute(
        text("SELECT 1 FROM auth.users WHERE email = :e"),
        {"e": body.email},
    ).fetchone()
    if exists:
        raise HTTPException(status_code=409, detail="Email already registered.")

    # validate requested roles exist, resolve to ids
    role_rows = session.execute(
        text("SELECT id, name FROM auth.roles WHERE name = ANY(:names)"),
        {"names": body.roles},
    ).fetchall()
    found = {r[1] for r in role_rows}
    missing = set(body.roles) - found
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown roles: {sorted(missing)}")

    try:
        pw_hash = hash_password(body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user_id = session.execute(
        text(
            """
            INSERT INTO auth.users (email, full_name, password_hash, auth_provider)
            VALUES (:e, :fn, :ph, 'local')
            RETURNING id
            """
        ),
        {"e": body.email, "fn": body.full_name, "ph": pw_hash},
    ).scalar_one()

    for role_id, _name in role_rows:
        session.execute(
            text(
                """
                INSERT INTO auth.user_roles (user_id, role_id)
                VALUES (:uid, :rid) ON CONFLICT DO NOTHING
                """
            ),
            {"uid": user_id, "rid": role_id},
        )
    session.commit()

    return MeOut(
        user_id=user_id,
        email=body.email,
        full_name=body.full_name,
        roles=_fetch_roles(session, user_id),
        permissions=_fetch_permissions(session, user_id),
    )


@router.post("/auth/login", response_model=TokenOut)
def login(
    body: LoginIn,
    request: Request,
    session: Session = Depends(get_db),
) -> TokenOut:
    """Authenticate an email/password user and issue access + refresh tokens.

    Uses one generic error for both a missing user and a wrong password to avoid
    user enumeration.

    Args:
        body: Login credentials.
        request: Incoming request, used to audit the issued refresh token.
        session: Active database session.

    Returns:
        Freshly issued access and refresh tokens.

    Raises:
        HTTPException: 401 on invalid credentials, 403 if the account is inactive.
    """
    row = session.execute(
        text(
            """
            SELECT id, password_hash, is_active
            FROM auth.users
            WHERE email = :e AND auth_provider = 'local'
            """
        ),
        {"e": body.email},
    ).fetchone()

    # same generic error for missing user vs wrong password (no user enumeration)
    if row is None or row[1] is None or not verify_password(body.password, row[1]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not row[2]:
        raise HTTPException(status_code=403, detail="Account is inactive.")

    return _issue_tokens(session, request, user_id=row[0], email=body.email)


@router.post("/auth/refresh", response_model=AccessOut)
def refresh(
    body: RefreshIn,
    session: Session = Depends(get_db),
) -> AccessOut:
    """Exchange a valid, unrevoked refresh token for a fresh access token.

    The refresh token is hashed and matched against a live row that is neither
    revoked nor expired; roles and permissions are re-read so the new access
    token reflects the user's current access.

    Args:
        body: The refresh token to redeem.
        session: Active database session.

    Returns:
        A new access token.

    Raises:
        HTTPException: 401 if the token is invalid or expired, 403 if the account
            is inactive.
    """
    token_hash = hash_refresh_token(body.refresh_token)
    row = session.execute(
        text(
            """
            SELECT rt.user_id, u.email, u.is_active
            FROM auth.refresh_tokens rt
            JOIN auth.users u ON u.id = rt.user_id
            WHERE rt.token_hash = :h
              AND rt.revoked_at IS NULL
              AND rt.expires_at > now()
            """
        ),
        {"h": token_hash},
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")
    if not row[2]:
        raise HTTPException(status_code=403, detail="Account is inactive.")

    user_id, email = row[0], row[1]
    roles = _fetch_roles(session, user_id)
    permissions = _fetch_permissions(session, user_id)
    access = create_access_token(user_id, email, roles, permissions)
    return AccessOut(access_token=access)


@router.post("/auth/logout")
def logout(
    body: LogoutIn,
    session: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> None:
    """Revoke a single refresh token, ending this device's session.

    Scoped to the caller's own tokens, so one user cannot revoke another's.

    Args:
        body: The refresh token to revoke.
        session: Active database session.
        user: Authenticated caller, used to scope the revocation.
    """
    session.execute(
        text(
            """
            UPDATE auth.refresh_tokens
            SET revoked_at = now()
            WHERE token_hash = :h AND user_id = :uid AND revoked_at IS NULL
            """
        ),
        {"h": hash_refresh_token(body.refresh_token), "uid": user.user_id},
    )
    session.commit()


@router.post("/auth/logout-all")
def logout_all(
    session: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> None:
    """Revoke every active session for the caller (logout from all devices).

    Args:
        session: Active database session.
        user: Authenticated caller whose sessions are being revoked.
    """
    session.execute(
        text(
            """
            UPDATE auth.refresh_tokens
            SET revoked_at = now()
            WHERE user_id = :uid AND revoked_at IS NULL
            """
        ),
        {"uid": user.user_id},
    )
    session.commit()


@router.get("/auth/me", response_model=MeOut)
def me(
    session: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> MeOut:
    """Return the authenticated caller's identity, roles, and permissions.

    Args:
        session: Active database session.
        user: Authenticated caller, resolved from the access token.

    Returns:
        The caller's id, email, name, roles, and permissions.
    """
    row = session.execute(
        text("SELECT full_name FROM auth.users WHERE id = :id"),
        {"id": user.user_id},
    ).fetchone()
    return MeOut(
        user_id=user.user_id,
        email=user.email,
        full_name=row[0] if row else None,
        roles=user.roles,
        permissions=user.permissions,
    )
