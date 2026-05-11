import logging
from typing import Optional, List, Tuple, Dict, Any
import uuid

from database.db_connection import get_pg_connection, return_pg_connection

logger = logging.getLogger(__name__)

def get_document_details_for_summary(document_id: str) -> Optional[Tuple[str, str, Optional[str]]]:
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.document_link, d."knowledgeBaseId", d.description
            FROM documents d
            WHERE d.id = %s AND d."isDeleted" = FALSE;
        """, (str(document_id),))
        result = cursor.fetchone()
        if result:
            return result[0], str(result[1]), result[2]
        return None
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin document (link, kb_id, description) {document_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            return_pg_connection(conn, pool)

def update_document_description_in_db(document_id: str, description: str) -> bool:
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE documents
            SET description = %s, updated_at = NOW()
            WHERE id = %s;
        """, (description, str(document_id)))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật description cho document {document_id}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_pg_connection(conn, pool)

def get_all_document_descriptions_for_kb(knowledge_base_id: str) -> List[str]:
    conn, pool = None, None
    descriptions: List[str] = []
    try:
        conn, pool = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.description
            FROM documents d
            WHERE d."knowledgeBaseId" = %s AND d."isDeleted" = FALSE AND d.description IS NOT NULL AND d.description <> '';
        """, (str(knowledge_base_id),))
        results = cursor.fetchall()
        for row in results:
            descriptions.append(row[0])
        return descriptions
    except Exception as e:
        logger.error(f"Lỗi khi lấy descriptions tài liệu cho knowledge_base {knowledge_base_id}: {e}", exc_info=True)
        return []
    finally:
        if conn:
            return_pg_connection(conn, pool)

def update_knowledge_base_description_in_db(knowledge_base_id: str, description: str) -> bool:
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE knowledge_base
            SET description = %s, updated_at = NOW()
            WHERE id = %s;
        """, (description, str(knowledge_base_id)))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật description cho knowledge_base {knowledge_base_id}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_pg_connection(conn, pool)

def get_knowledge_base_current_description(knowledge_base_id: str) -> Optional[str]:
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT description
            FROM knowledge_base
            WHERE id = %s;
        """, (str(knowledge_base_id),))
        result = cursor.fetchone()
        if result:
            return result[0]
        return None
    except Exception as e:
        logger.error(f"Lỗi khi lấy mô tả hiện tại của knowledge_base {knowledge_base_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            return_pg_connection(conn, pool) 