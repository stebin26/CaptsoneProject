# services/api-gateway/api_app/auth/bootstrap.py
"""One-off CLI to create or promote an Admin user.

Breaks the admin chicken-and-egg problem: the first Admin cannot be created
through the permission-guarded register endpoint, so this script seeds one
directly against the database.

Example:
    docker compose exec api python -m api_app.auth.bootstrap
    --email admin@ops.local --password "ChangeMe123" --name "Platform Admin"
"""

from __future__ import annotations

import argparse
import sys

from ops_common.db import session_scope
from sqlalchemy import text

from api_app.auth.passwords import hash_password


def create_admin(email: str, password: str, full_name: str | None) -> None:
    """Create a new Admin user, or promote an existing user to Admin.

    If the email already exists, the account is reset to a known-good state
    (password reset, marked active, provider set to local) and guaranteed the
    Admin role. Otherwise a new local user is created and granted Admin. Exits
    the process with a message if the password is rejected or the Admin role is
    missing from the schema.

    Args:
        email: Email address of the admin user to create or promote.
        password: Plaintext password to set (subject to the hashing policy).
        full_name: Optional display name for a newly created user.
    """
    try:
        pw_hash = hash_password(password)
    except ValueError as exc:
        sys.exit(f"Password rejected: {exc}")

    with session_scope() as s:
        admin_role = s.execute(
            text("SELECT id FROM auth.roles WHERE name = 'Admin'")
        ).fetchone()
        if admin_role is None:
            sys.exit("Admin role not found — is auth_schema.sql applied?")
        admin_role_id = admin_role[0]

        existing = s.execute(
            text("SELECT id FROM auth.users WHERE email = :e"), {"e": email}
        ).fetchone()

        if existing:
            user_id = existing[0]
            # promote: reset password, ensure active + local, guarantee Admin role
            s.execute(
                text(
                    """
                    UPDATE auth.users
                    SET password_hash = :ph, auth_provider = 'local',
                        is_active = TRUE
                    WHERE id = :id
                    """
                ),
                {"ph": pw_hash, "id": user_id},
            )
            action = "updated (password reset, ensured Admin)"
        else:
            user_id = s.execute(
                text(
                    """
                    INSERT INTO auth.users
                        (email, full_name, password_hash, auth_provider)
                    VALUES (:e, :fn, :ph, 'local')
                    RETURNING id
                    """
                ),
                {"e": email, "fn": full_name, "ph": pw_hash},
            ).scalar_one()
            action = "created"

        s.execute(
            text(
                """
                INSERT INTO auth.user_roles (user_id, role_id)
                VALUES (:uid, :rid) ON CONFLICT DO NOTHING
                """
            ),
            {"uid": user_id, "rid": admin_role_id},
        )

    print(f"Admin {action}: {email} (id={user_id})")


def main() -> None:
    """Parse CLI arguments and run the admin create-or-promote flow."""
    parser = argparse.ArgumentParser(description="Create or promote an Admin user.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    create_admin(args.email, args.password, args.name)


if __name__ == "__main__":
    main()
