import os
import secrets
import hashlib
import datetime
from datetime import timedelta
from typing import Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from backend.app.core.database import get_db_connection

JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key-change-in-prod-1234567890")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

security_scheme = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    """
    Hashes a plaintext password using bcrypt directly.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a bcrypt hash directly.
    """
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def hash_refresh_token(token: str) -> str:
    """
    Hashes a refresh token with SHA-256 for secure database storage.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def create_access_token(user_id: str, username: str) -> str:
    """
    Generates a signed HS256 JWT access token.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "user_id": user_id,
        "exp": int(expire.timestamp())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def create_refresh_token(user_id: str, db_conn = None) -> str:
    """
    Generates a secure refresh token, inserts its SHA-256 hash into SQLite,
    and returns the raw token to the client. Reuses db_conn if provided to prevent deadlocks.
    """
    token = secrets.token_hex(32)
    token_hash = hash_refresh_token(token)
    
    now = datetime.datetime.now(datetime.timezone.utc)
    expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expire_str = expire.isoformat()
    
    conn = db_conn if db_conn is not None else get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO refresh_tokens (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expire_str)
        )
        if db_conn is None:
            conn.commit()
    finally:
        if db_conn is None:
            conn.close()
        
    return token

def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes and validates a JWT access token. Raises 401 if invalid/expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)) -> dict:
    """
    FastAPI dependency to secure routes. Requires Authorization header containing valid JWT.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user information",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"id": row["id"], "username": row["username"]}
    finally:
        conn.close()
