from fastapi import APIRouter, HTTPException, Query, Body, Depends, Request, Response
from typing import List, Dict, Any, Optional, Callable
import uuid
import time
import logging
from services.query_service import QueryService
from database.query_config_db import QueryConfigDB
from services.conversation_manager import conversation_manager
from models.config_models import (
    QueryRequest, QueryResponse, QueryConfigCreate,
    QueryConfigUpdate, QueryConfigResponse
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Knowledge Base Query"])


# Helper function for request ID
def get_request_id(request: Request) -> str:
    """Get or create request ID from request headers"""
    if "X-Request-ID" in request.headers:
        return request.headers["X-Request-ID"]
    return str(uuid.uuid4())


# You'll need to add this middleware to your main FastAPI app instead
def create_middleware():
    """
    Creates a middleware that can be added to the main FastAPI app
    Example:
        app = FastAPI()
        app.middleware("http")(chunks_search_api.create_middleware())
    """

    async def add_request_id_middleware(request: Request, call_next):
        request_id = get_request_id(request)
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request {request_id}: Completed in {process_time:.2f}s with status {response.status_code}")

        return response

    return add_request_id_middleware

# Thêm hàm kiểm tra UUID
def is_valid_uuid(uuid_string):
    try:
        uuid_obj = uuid.UUID(uuid_string)
        return str(uuid_obj) == uuid_string
    except:
        return False

# Add dependency for logging
async def log_request(request: Request):
    request_id = get_request_id(request)
    logger.info(f"Processing request {request_id}: {request.method} {request.url.path}")
    return request_id


@router.post("/query/stream/")
async def query_document_stream(request: QueryRequest, request_id: str = Depends(log_request)):
    """
    API streaming cho RAG - nhận câu hỏi, tìm chunks liên quan và stream câu trả lời từ LLM.
    Trả về Server-Sent Events (SSE) với thông tin chi tiết về quá trình RAG.
    
    Response format:
    - Start: {"started": true, "config_name": "...", "model": "..."}
    - Status updates: {"status": "rag_processing", "stage": "embedding|search|chunks_found|llm_generating"}
    - Chunks info: {"status": "chunks_info", "chunks": [...]}
    - Streaming tokens: {"token": "text", "finished": false}
    - End: {"finished": true, "metadata": {...}}
    """
    from fastapi.responses import StreamingResponse
    import asyncio
    import json
    
    start_time = time.time()
    
    logger.info(f"Request {request_id}: Processing STREAMING RAG query: {request.query}")
    
    # Log chat_section_id if provided
    if request.chat_section_id:
        logger.info(f"Streaming RAG with chat_section_id: {request.chat_section_id}")
    
    async def generate_rag_stream():
        try:
            # Stream from QueryService.execute_query_stream
            async for chunk in QueryService.execute_query_stream(request, chat_section_id=request.chat_section_id):
                yield chunk
                
        except ValueError as e:
            logger.error(f"Request {request_id}: ValueError in streaming RAG: {str(e)}")
            yield f"data: {json.dumps({'error': f'Lỗi cấu hình: {str(e)}', 'finished': True})}\n\n"
        except Exception as e:
            logger.error(f"Request {request_id}: Error in streaming RAG: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Lỗi nội bộ: {str(e)}', 'finished': True})}\n\n"
    
    # Return StreamingResponse with optimized SSE headers for smooth streaming
    return StreamingResponse(
        generate_rag_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Content-Type-Options": "nosniff",
            "Transfer-Encoding": "chunked",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "X-Request-ID",
            "X-Request-ID": request_id,
        }
    )


@router.post("/query/", response_model=QueryResponse)
async def query_document(request: QueryRequest, request_id: str = Depends(log_request)):
    """
    API nhận câu hỏi từ người dùng, tìm các chunk liên quan, rồi gửi đến LLM để sinh câu trả lời.
    Sử dụng cấu hình từ database.
    
    - Nếu cung cấp chat_section_id, hệ thống sẽ sử dụng memory của cuộc hội thoại đó để cải thiện câu trả lời
    và tự động cập nhật memory sau mỗi lần tương tác.
    """
    try:
        # Log thông tin chat_section_id nếu có
        if request.chat_section_id:
            logger.info(f"Query with chat_section_id: {request.chat_section_id}")
        
        # Truyền chat_section_id vào execute_query để sử dụng và cập nhật memory
        result = await QueryService.execute_query(request, chat_section_id=request.chat_section_id)
        result["request_id"] = uuid.UUID(request_id)  # Add request_id to response
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {str(e)}")


# Endpoints CRUD cho query_config
@router.post("/configs/", response_model=Dict[str, Any])
async def create_query_config(config: QueryConfigCreate, request_id: str = Depends(log_request)):
    """Tạo cấu hình truy vấn mới"""
    logger.debug(f"Received config: {config}")
    try:
        config_id = await QueryConfigDB.create_config(config)
        return {"id": config_id, "message": "Tạo cấu hình thành công"}
    except Exception as e:
        logger.error(f"Error creating config: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {str(e)}")


@router.get("/configs/{config_id}", response_model=Dict[str, Any])
async def get_query_config(config_id: str, request_id: str = Depends(log_request)):
    """Lấy thông tin cấu hình theo ID"""
    if not is_valid_uuid(config_id):
        return "Lỗi id không hợp lệ"
    try:
        config = await QueryConfigDB.get_config(config_id)
        if not config:
            raise HTTPException(status_code=404, detail="Không tìm thấy cấu hình")
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config {config_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {str(e)}")


@router.get("/configs/by-knowledge-base/{knowledge_base_id}", response_model=List[Dict[str, Any]])
async def get_configs_by_knowledge_base(knowledge_base_id: uuid.UUID, request_id: str = Depends(log_request)):
    """Lấy tất cả cấu hình của một knowledge base"""
    logger.info(f"Getting configs for knowledge base: {knowledge_base_id}")
    try:
        configs = await QueryConfigDB.get_configs_by_knowledge_base(knowledge_base_id)
        logger.info(f"Found {len(configs)} configs")
        return configs
    except Exception as e:
        logger.error(f"Error getting configs for KB {knowledge_base_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {str(e)}")


@router.put("/configs/{config_id}", response_model=Dict[str, Any])
async def update_query_config(config_id: uuid.UUID, update_data: QueryConfigUpdate,
                              request_id: str = Depends(log_request)):
    """Cập nhật thông tin cấu hình"""
    try:
        success = await QueryConfigDB.update_config(config_id, update_data)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy cấu hình hoặc không có thay đổi")
        return {"message": "Cập nhật cấu hình thành công"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config {config_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {str(e)}")


@router.delete("/configs/{config_id}", response_model=Dict[str, Any])
async def delete_query_config(config_id: uuid.UUID, request_id: str = Depends(log_request)):
    """Xóa mềm cấu hình"""
    try:
        success = await QueryConfigDB.delete_config(config_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy cấu hình")
        return {"message": "Xóa cấu hình thành công"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting config {config_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {str(e)}")


@router.get("/configs/default/{knowledge_base_id}", response_model=Dict[str, Any])
async def get_default_config(knowledge_base_id: uuid.UUID, request_id: str = Depends(log_request)):
    """Lấy cấu hình mặc định của một knowledge base"""
    try:
        config = await QueryConfigDB.get_default_config(knowledge_base_id)
        if not config:
            raise HTTPException(status_code=404, detail="Không tìm thấy cấu hình mặc định")
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting default config for KB {knowledge_base_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {str(e)}")


# Conversation Management APIs
@router.get("/conversation/{chat_section_id}/history/")
async def get_conversation_history(chat_section_id: str):
    """Lấy conversation history cho chat section"""
    try:
        history = conversation_manager.get_conversation_history(chat_section_id)
        stats = conversation_manager.get_conversation_stats(chat_section_id)
        
        return {
            "status": "success",
            "chat_section_id": chat_section_id,
            "history": history,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{chat_section_id}/")
async def clear_conversation(chat_section_id: str):
    """Xóa conversation history"""
    try:
        conversation_manager.clear_conversation(chat_section_id)
        return {
            "status": "success",
            "message": f"Đã xóa conversation history cho {chat_section_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/stats/")
async def get_conversation_stats():
    """Lấy thống kê tổng quan về conversations"""
    try:
        # Cleanup old conversations
        conversation_manager.cleanup_old_conversations()
        
        return {
            "status": "success",
            "message": "Conversation stats retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting conversation stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
