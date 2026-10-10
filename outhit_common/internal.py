"""Trust between services.

Only the gateway faces the internet. It verifies the user's JWT and forwards the
request with two headers:
  X-Internal-Token  shared secret proving the call came from inside Outhit
  X-User-Id         the authenticated user's id (absent for anonymous requests)
Services bind to 127.0.0.1 and reject user-scoped calls without a valid token.
"""

import hmac
import os

from fastapi import Header, HTTPException, status


def internal_token() -> str:
    return os.environ.get("INTERNAL_TOKEN", "dev-internal-token-change-me")


def _check(token: str | None) -> None:
    if not token or not hmac.compare_digest(token, internal_token()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This endpoint is only reachable through the Outhit gateway.")


def optional_user(x_internal_token: str | None = Header(default=None), x_user_id: str | None = Header(default=None)) -> int | None:
    if x_user_id is None:
        return None
    _check(x_internal_token)
    return int(x_user_id)


def require_user(x_internal_token: str | None = Header(default=None), x_user_id: str | None = Header(default=None)) -> int:
    _check(x_internal_token)
    if x_user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return int(x_user_id)


def require_internal(x_internal_token: str | None = Header(default=None)) -> None:
    _check(x_internal_token)
