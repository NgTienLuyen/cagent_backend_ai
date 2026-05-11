"""
Utilities for model management
"""
import os
from database.db_connection import get_pg_connection, return_pg_connection
import uuid
from typing import Optional, Dict, Any
import logging

# Constants
LM_STUDIO_ENDPOINT = os.getenv("LM_STUDIO_ENDPOINT", "http://127.0.0.1:1234/v1")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # API Key mặc định

logger = logging.getLogger(__name__)


def determine_category_from_name(model_name):
    """Xác định category từ tên model"""
    if "GPT" in model_name or "text-embedding" in model_name or "ada" in model_name:
        return "openai"
    elif "Gemini" in model_name:
        return "google"
    elif "Cohere" in model_name or "Command" in model_name or "embed-" in model_name:
        return "cohere"
    elif "vistral" in model_name.lower():
        return "local"
    return None


def get_api_key_for_model(model_id):
    """Lấy API key cho model từ DB hoặc environment"""
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            # Lấy model và category info
            cursor.execute(
                """
                SELECT m.api_key, c.api_type 
                FROM models m
                JOIN model_categories c ON m.category_id = c.id
                WHERE m.id = %s
                """,
                (str(model_id),)
            )

            result = cursor.fetchone()
            return dict(zip([col[0] for col in cursor.description], result))
    except Exception as e:
        logger.error(f"Lỗi khi lấy API key cho model {model_id}: {e}")
        return None
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


def get_api_key_by_category(category_code):
    """Lấy API key cho category từ DB hoặc environment"""
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            # Lấy api_type từ category
            cursor.execute(
                "SELECT api_type FROM model_categories WHERE code_name = %s",
                (category_code,)
            )

            result = cursor.fetchone()
            return dict(zip([col[0] for col in cursor.description], result))
    except Exception as e:
        logger.error(f"Lỗi khi lấy API key cho category {category_code}: {e}")
        return None
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


def get_api_key_by_type(api_type):
    """Lấy API key dựa trên loại API"""
    if api_type == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif api_type == "google":
        return os.getenv("GEMINI_API_KEY")
    elif api_type == "cohere":
        return os.getenv("COHERE_API_KEY")
    return None


async def find_model_by_name(model_name):
    """Tìm model theo tên hoặc code_name"""
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id FROM models 
                WHERE (name = %s OR code_name = %s) 
                AND status = TRUE AND is_deleted = FALSE
                LIMIT 1
                """,
                (model_name, model_name)
            )

            result = cursor.fetchone()
            return uuid.UUID(result[0]) if result else None
    except Exception as e:
        logger.error(f"Lỗi khi tìm model bởi tên {model_name}: {e}")
        return None
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


async def find_suitable_embedding_model(category_code):
    """Tìm model embedding phù hợp cho danh mục"""
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT m.id FROM models m
                JOIN model_categories c ON m.category_id = c.id
                WHERE c.code_name = %s 
                AND m.capability->>'embedding' = 'true'
                AND m.status = TRUE AND m.is_deleted = FALSE
                LIMIT 1
                """,
                (category_code,)
            )

            result = cursor.fetchone()
            return uuid.UUID(result[0]) if result else None
    except Exception as e:
        logger.error(f"Lỗi khi tìm model embedding phù hợp cho danh mục {category_code}: {e}")
        return None
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)