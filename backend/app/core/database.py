import os
import sqlite3

DATABASE_PATH = os.getenv("SQLITE_DATABASE_PATH", "backend/data/repomindai.db")

def get_db_connection() -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.
    Configures row factory to return dict-like objects and enables WAL mode for concurrency.
    """
    dir_name = os.path.dirname(DATABASE_PATH)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Enable Write-Ahead Log (WAL) mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    """
    Initializes tables if they do not exist.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        
        # Create refresh_tokens table (stores SHA-256 hashed refresh tokens)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
        
        conn.commit()
    finally:
        conn.close()
