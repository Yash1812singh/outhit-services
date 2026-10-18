"""User accounts and JWTs. The gateway is the only service that knows about passwords.

Sign-in is by email + password or with Google (the browser gets an ID token from
Google Identity Services and we verify its signature, audience and issuer here).
Every token carries a jti; logging out records it as revoked until it expires.
"""

import secrets
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import DateTime, String, create_engine, delete, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from gateway.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100), default="")
    password_hash: Mapped[str] = mapped_column(String(255))  # "" for Google-only accounts
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True, index=True, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


def init_db() -> None:
    """Create tables and add columns introduced after the first release (SQLite/Postgres safe)."""
    Base.metadata.create_all(bind=engine)
    columns = {c["name"] for c in inspect(engine).get_columns("users")}
    if "google_sub" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN google_sub VARCHAR(255)"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_sub ON users (google_sub)"))


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RegisterRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(default="", max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleLoginRequest(BaseModel):
    credential: str = Field(min_length=20, max_length=8192)


class UserOut(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str
    has_password: bool = True
    google_linked: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_token(user_id: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode({"sub": str(user_id), "exp": exp, "jti": secrets.token_urlsafe(16)}, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _claims(request: Request) -> dict | None:
    """Claims of a valid, unrevoked Bearer token, or None (missing, expired, forged or logged out)."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    try:
        payload = jwt.decode(header[7:], settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
    jti = payload.get("jti")
    if jti:
        with SessionLocal() as db:
            if db.get(RevokedToken, jti) is not None:
                return None
    return payload


def user_id_from_request(request: Request) -> int | None:
    """The user id from a valid Bearer token, or None (missing, expired, forged or logged out)."""
    claims = _claims(request)
    return int(claims["sub"]) if claims else None


# Google Identity Services ID tokens are RS256 JWTs signed with these rotating keys.
GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")
_google_keys = jwt.PyJWKClient(GOOGLE_CERTS_URL, cache_keys=True, lifespan=3600)


def verify_google_token(credential: str) -> dict:
    """Verified claims of a Google ID token. Raises jwt.PyJWTError if it is not genuine and meant for us."""
    key = _google_keys.get_signing_key_from_jwt(credential).key
    claims = jwt.decode(credential, key, algorithms=["RS256"], audience=settings.google_client_id, options={"require": ["exp", "iss", "sub", "aud"]})
    if claims.get("iss") not in GOOGLE_ISSUERS:
        raise jwt.InvalidIssuerError("Token was not issued by Google")
    return claims


def _out(u: User) -> UserOut:
    return UserOut(id=u.id, email=u.email, first_name=u.first_name, last_name=u.last_name,
                   has_password=bool(u.password_hash), google_linked=bool(u.google_sub))


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(email=email, first_name=body.first_name.strip(), last_name=body.last_name.strip(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return TokenResponse(access_token=create_token(user.id), user=_out(user))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user and not user.password_hash and user.google_sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "This account uses Google sign-in. Use the \"Sign in with Google\" button.")
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return TokenResponse(access_token=create_token(user.id), user=_out(user))


@router.get("/config")
def config() -> dict:
    """Public sign-in options for the frontend."""
    return {"google_client_id": settings.google_client_id or None}


@router.post("/google", response_model=TokenResponse)
def google(body: GoogleLoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Sign in (or sign up) with a Google ID token. Links to an existing account with the same verified email."""
    if not settings.google_client_id:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Google sign-in is not configured on this server.")
    try:
        claims = verify_google_token(body.credential)
    except (jwt.PyJWTError, jwt.PyJWKClientError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Google sign-in failed. Please try again.") from exc
    email = str(claims.get("email", "")).lower()
    if not email or claims.get("email_verified") not in (True, "true"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your Google account's email is not verified.")

    user = db.scalar(select(User).where(User.google_sub == claims["sub"]))
    if user is None:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, first_name=(claims.get("given_name") or email.split("@")[0])[:100],
                        last_name=(claims.get("family_name") or "")[:100], password_hash="")
            db.add(user)
        user.google_sub = claims["sub"]
        db.commit()
    return TokenResponse(access_token=create_token(user.id), user=_out(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, db: Session = Depends(get_db)) -> None:
    """Revoke the current token. Always succeeds, so a client can log out even with an expired token."""
    claims = _claims(request)
    now = datetime.now(timezone.utc)
    db.execute(delete(RevokedToken).where(RevokedToken.expires_at < now))  # keep the table small
    if claims and claims.get("jti"):
        db.add(RevokedToken(jti=claims["jti"], expires_at=datetime.fromtimestamp(claims["exp"], timezone.utc)))
    db.commit()


@router.get("/me", response_model=UserOut)
def me(request: Request, db: Session = Depends(get_db)) -> UserOut:
    uid = user_id_from_request(request)
    user = db.get(User, uid) if uid else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return _out(user)
