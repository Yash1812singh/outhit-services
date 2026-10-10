"""Encrypts broker secrets at rest. The key lives only in the environment (BROKER_ENCRYPTION_KEY)."""

from cryptography.fernet import Fernet, InvalidToken

from broker.config import get_settings


class VaultError(RuntimeError):
    pass


def _fernet() -> Fernet:
    key = get_settings().encryption_key
    if not key:
        raise VaultError("BROKER_ENCRYPTION_KEY is not set; broker connections are disabled.")
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as exc:
        raise VaultError("BROKER_ENCRYPTION_KEY is not a valid Fernet key.") from exc


def seal(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def open_(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return _fernet().decrypt(value.encode()).decode()
    except InvalidToken as exc:
        raise VaultError("Stored broker credentials can't be decrypted (was the encryption key changed?). Reconnect.") from exc


def mask(value: str | None) -> str | None:
    return f"{value[:4]}…{value[-4:]}" if value and len(value) > 10 else ("set" if value else None)
