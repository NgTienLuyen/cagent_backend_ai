import os
import psycopg2
import asyncpg
import logging
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool
import threading

# Load environment variables if .env file exists
try:
    load_dotenv()
except Exception as e:
    pass

logger = logging.getLogger(__name__)

# Default configuration with environment variable fallbacks
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "ai_agent"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "newpassword"),
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": os.getenv("DB_PORT", "5432")
}

# Connection pool for async operations
_pool = None

# Connection pool for synchronous operations
_sync_pool = None
_sync_pool_lock = threading.Lock()

def get_sync_pool():
    """Get synchronous database connection pool"""
    global _sync_pool
    if _sync_pool is None:
        with _sync_pool_lock:
            if _sync_pool is None:
                try:
                    _sync_pool = SimpleConnectionPool(
                        minconn=10,
                        maxconn=50,
                        **DB_CONFIG
                    )
                    logger.info("Synchronous database connection pool created")
                except Exception as e:
                    logger.error(f"Error creating sync pool: {str(e)}")
                    raise
    return _sync_pool

def get_pg_connection():
    """
    Lấy một kết nối đồng bộ từ pool.
    Trả về một tuple (connection, pool).
    """
    try:
        pool = get_sync_pool()
        conn = pool.getconn()
        register_vector(conn)
        logger.debug("Got connection from sync pool")
        return conn, pool
    except Exception as e:
        logger.error(f"Error getting database connection from pool: {str(e)}")
        raise

def return_pg_connection(conn, pool):
    """Trả một kết nối đồng bộ về lại pool."""
    if pool and conn:
        pool.putconn(conn)
        logger.debug("Returned connection to sync pool")

async def get_db_pool():
    """Get database connection pool for async operations"""
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                database=DB_CONFIG["dbname"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"],
                min_size=20,
                max_size=100,
                command_timeout=60
            )
            async with _pool.acquire() as conn:
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            logger.info("Async database connection pool created with optimized settings")
        except Exception as e:
            logger.error(f"Error creating async database pool: {str(e)}")
            raise
    return _pool

async def close_db_pool():
    """Close all database connection pools"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Async database connection pool closed")
    
    global _sync_pool
    if _sync_pool:
        _sync_pool.closeall()
        _sync_pool = None
        logger.info("Synchronous database connection pool closed")
