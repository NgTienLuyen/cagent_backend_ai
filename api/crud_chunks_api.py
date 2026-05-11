from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import List
from uuid import UUID
from datetime import datetime
from database.db_connection import get_pg_connection, return_pg_connection  # Sửa ở đây
from pgvector.psycopg2 import register_vector

# Khởi tạo router cho các API CRUD
router = APIRouter()

# Mô hình dữ liệu Chunk
class Chunk(BaseModel):
    chunk_text: str
    document_id: UUID

class UpdateChunk(BaseModel):
    chunk_text: str


# Thêm một chunk mới
@router.post("/chunks/")
async def create_chunk(chunk: Chunk):
    """
    Thêm một chunk mới vào cơ sở dữ liệu.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            register_vector(cursor)

            # Insert chunk vào bảng với đầy đủ thông tin cần thiết
            # Chú ý: Sử dụng tên cột chính xác theo cấu trúc bảng
            cursor.execute("""
                INSERT INTO chunks ("document_id", "chunk_text", "isDelete", "createTime", "updated_at", "isembeddingdone")
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                str(chunk.document_id),
                chunk.chunk_text,
                False,
                datetime.utcnow(),
                datetime.utcnow(),
                False
            ))

            chunk_id = cursor.fetchone()[0]

            # Cập nhật trạng thái isChunked của tài liệu thành true
            cursor.execute("""
                UPDATE documents
                SET "isChunked" = true, "updated_at" = %s
                WHERE "id" = %s;
            """, (datetime.utcnow(), str(chunk.document_id)))

            conn.commit()
        return {"message": "Chunk created successfully", "chunk_id": chunk_id, "document_id": chunk.document_id}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating chunk: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)


# Sửa một chunk
@router.put("/chunks/{chunk_id}")
async def update_chunk(chunk_id: UUID, update_chunk: UpdateChunk):
    """
    Cập nhật thông tin chunk.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:

            # Update chunk
            cursor.execute("""
                UPDATE chunks
                SET "chunk_text" = %s, "updated_at" = %s
                WHERE "id" = %s AND "isDelete" = false
                RETURNING "id";
            """, (update_chunk.chunk_text, datetime.utcnow(), str(chunk_id)))

            updated_chunk_id = cursor.fetchone()

            if not updated_chunk_id:
                raise HTTPException(status_code=404, detail="Chunk not found or is deleted")

            conn.commit()
        return {"message": "Chunk updated successfully", "chunk_id": chunk_id}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating chunk: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)


# Xóa mềm một chunk (đánh dấu là đã xóa, nhưng không thực sự xóa khỏi DB)
@router.patch("/chunks/{chunk_id}/soft_delete")
async def soft_delete_chunk(chunk_id: UUID):
    """
    Xóa mềm một chunk, đánh dấu là đã xóa.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:

            # Update trường isDeleted thành True (xóa mềm)
            cursor.execute("""
                UPDATE chunks
                SET "isDelete" = true, "deleteTime" = %s
                WHERE "id" = %s AND "isDelete" = false
                RETURNING "id";
            """, (datetime.utcnow(), str(chunk_id)))

            deleted_chunk_id = cursor.fetchone()

            if not deleted_chunk_id:
                raise HTTPException(status_code=404, detail="Chunk not found or already deleted")

            conn.commit()
        return {"message": "Chunk soft deleted successfully", "chunk_id": chunk_id}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error soft deleting chunk: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)


# Xóa vĩnh viễn một chunk (thực sự xóa khỏi DB)
@router.delete("/chunks/{chunk_id}")
async def hard_delete_chunk(chunk_id: UUID):
    """
    Xóa vĩnh viễn một chunk khỏi cơ sở dữ liệu.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:

            # Thực hiện xóa vĩnh viễn
            cursor.execute("""
                DELETE FROM chunks
                WHERE "id" = %s 
                RETURNING "id";
            """, (str(chunk_id),))

            deleted_chunk_id = cursor.fetchone()

            if not deleted_chunk_id:
                raise HTTPException(status_code=404, detail="Chunk not found or not soft deleted")

            conn.commit()
        return {"message": "Chunk hard deleted successfully", "chunk_id": chunk_id}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error hard deleting chunk: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)

@router.patch("/chunks/{chunk_id}/restore")
async def restore_chunk(chunk_id: UUID):
    """
    Khôi phục một chunk đã bị xóa mềm (isDelete=true) về trạng thái bình thường (isDelete=false).
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:

            # Đặt lại isDelete = false
            cursor.execute("""
                UPDATE chunks
                SET "isDelete" = false, "updated_at" = %s
                WHERE "id" = %s AND "isDelete" = true
                RETURNING "id";
            """, (datetime.utcnow(), str(chunk_id)))

            restored_chunk_id = cursor.fetchone()

            if not restored_chunk_id:
                raise HTTPException(
                    status_code=404,
                    detail="Chunk not found or it wasn't soft-deleted"
                )

            conn.commit()
        return {"message": "Chunk restored successfully", "chunk_id": chunk_id}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error restoring chunk: {str(e)}"
        )
    finally:
        if conn and pool: return_pg_connection(conn, pool)


# Lấy danh sách các chunk không bị xóa mềm
@router.get("/chunks/")
async def get_chunks():
    """
    Lấy danh sách các chunk chưa bị xóa.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:

            cursor.execute("""
                SELECT "id", "document_id", "chunk_text", "createTime", "updated_at", "isDelete"
                FROM chunks;
            """)

            chunks = cursor.fetchall()
            conn.commit()
        return {"chunks": [{"id": chunk[0], "document_id": chunk[1], "chunk_text": chunk[2],"createTime": chunk[3], "updated_at": chunk[4], "isDelete":chunk[5] } for chunk in chunks]}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error fetching chunks: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)


# Lấy danh sách các chunk theo document_id
@router.get("/chunks/{document_id}")
async def get_chunks_by_document_id(document_id: str):
    """
    API lấy danh sách các chunk từ bảng chunks theo document_id.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            register_vector(cursor)

            # Truy vấn các chunk theo document_id
            cursor.execute("""
                SELECT "id", "document_id", "chunk_text", "createTime", "updated_at", "isDelete", "isembeddingdone"
                FROM chunks
                WHERE "document_id" = %s 
                ORDER BY "createTime";
            """, (document_id,))

            # Lấy kết quả trả về
            rows = cursor.fetchall()

            # Kiểm tra nếu không có dữ liệu trả về
            if not rows:
                return {"document_id": document_id, "chunks": []} # Trả về mảng rỗng thay vì lỗi 404

            # Trả về danh sách các chunk
            chunks = [{"id": row[0], "document_id": row[1], "chunk_text": row[2],
                       "createTime": row[3], "updated_at": row[4], "isDelete": row[5], "isEmbeddingDone": row[6]} for row in rows]

            # Đóng kết nối cơ sở dữ liệu
        return {"document_id": document_id, "chunks": chunks}

    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error fetching chunks: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)


@router.get("/chunks/{chunk_id}")
async def get_chunk_by_id(chunk_id: str):
    """
    API lấy một chunk theo id từ bảng chunks.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()

        # Ensure cursor and connection are properly closed after use
        with conn.cursor() as cursor:
            # Register pgvector extension
            register_vector(cursor)

            # Execute query to fetch chunk by id
            cursor.execute("""
                SELECT "id", "document_id", "chunk_text", "createTime", "updated_at", "isDelete"
                FROM chunks
                WHERE "id" = %s AND "isDelete" = false
            """, (chunk_id,))

            # Fetch result
            row = cursor.fetchone()

            # If no data is found, raise 404 error
            if not row:
                raise HTTPException(status_code=404, detail=f"Chunk with id {chunk_id} not found")

            # Construct the chunk response
            chunk = {
                "id": row[0],
                "document_id": row[1],
                "chunk_text": row[2],
                "createTime": row[3],
                "updated_at": row[4],
                "isDelete": row[5]
            }

        # Return the chunk data
        return {"chunk": chunk}

    except Exception as e:
        if conn: conn.rollback()
        # Catch any errors and return as HTTPException
        raise HTTPException(status_code=500, detail=f"Error fetching chunk: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)


# API hủy bỏ embedding của tài liệu
@router.patch("/documents/{document_id}/reset-embedding")
async def reset_document_embedding(document_id: UUID):
    """
    Hủy bỏ embedding của tài liệu - chuyển trường isEmbeddingDone về false và xóa dữ liệu embedding.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            register_vector(cursor)  # Đảm bảo đăng ký pgvector extension

            # Cập nhật trường isEmbeddingDone về false và đặt embedding về NULL
            cursor.execute("""
                UPDATE documents
                SET "isEmbeddingDone" = false, "updated_at" = %s
                WHERE "id" = %s
                RETURNING "id";
            """, (datetime.utcnow(), str(document_id)))

            updated_doc = cursor.fetchone()

            if not updated_doc:
                raise HTTPException(status_code=404, detail="Document not found")

            # Cập nhật trường embedding về NULL trong bảng chunks
            cursor.execute("""
                UPDATE chunks
                SET "embedding" = NULL, "isembeddingdone" = false, "updated_at" = %s
                WHERE "document_id" = %s;
            """, (datetime.utcnow(), str(document_id)))

            conn.commit()
        return {"message": "Document embedding reset successfully", "document_id": document_id}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error resetting document embedding: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)


# API xóa toàn bộ chunks của tài liệu và reset trạng thái
@router.delete("/documents/{document_id}/chunks")
async def delete_document_chunks(document_id: UUID):
    """
    Xóa toàn bộ chunks của tài liệu và chuyển trường isChunked và isEmbeddingDone về false.
    """
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:

            # Bắt đầu transaction
            cursor.execute("BEGIN;")

            # 1. Xóa toàn bộ chunks thuộc về document_id
            cursor.execute("""
                DELETE FROM chunks
                WHERE "document_id" = %s
                RETURNING id;
            """, (str(document_id),))

            deleted_chunks = cursor.fetchall()

            # 2. Cập nhật document: đặt isChunked và isEmbeddingDone về false
            cursor.execute("""
                UPDATE documents
                SET "isChunked" = false, "isEmbeddingDone" = false, "updated_at" = %s
                WHERE "id" = %s
                RETURNING "id";
            """, (datetime.utcnow(), str(document_id)))

            updated_doc = cursor.fetchone()

            if not updated_doc:
                # Nếu không tìm thấy document, rollback và báo lỗi
                cursor.execute("ROLLBACK;")
                raise HTTPException(status_code=404, detail="Document not found")

            # Commit transaction nếu mọi thứ thành công
            cursor.execute("COMMIT;")

            # Đóng kết nối
        return {
            "message": "Document chunks deleted and status reset successfully",
            "document_id": document_id,
            "deleted_chunks_count": len(deleted_chunks)
        }
    except Exception as e:
        if conn: conn.rollback()
        # Đảm bảo rollback nếu có lỗi
        try:
            cursor.execute("ROLLBACK;")
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error deleting document chunks: {str(e)}")
    finally:
        if conn and pool: return_pg_connection(conn, pool)
