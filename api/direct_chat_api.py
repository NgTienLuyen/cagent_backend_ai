from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, UUID4
from typing import Dict, Any, Optional, List
import uuid
import time
import logging
from datetime import datetime
from database.db_connection import get_pg_connection, return_pg_connection
from llms.onlinellms import OnlineLLMs
from llms.localLllms import LocalLlms
from models.config_models import QueryResponse, ChunkResponse
from services.query_service import QueryService
from llms.config_loader import PromptBuilderLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Direct LLM Chat"])


class DirectLLMRequest(BaseModel):
    """
    Request model cho direct LLM chat API.
    Đơn giản hơn QueryRequest, chỉ cần query và model_name.
    """
    query: str
    model_name: str = "GPT-4"  # Mặc định là GPT-4
    model_type: str = "online" # "online" hoặc "local"
    api_key: Optional[str] = None  # API key tùy chỉnh
    endpoint: Optional[str] = None  # Endpoint tùy chỉnh cho model
    parameters: Optional[Dict[str, Any]] = None  # Tham số bổ sung như temperature, max_tokens...
    chat_section_id: Optional[UUID4] = None  # ID của chat section để quản lý memory

    class Config:
        schema_extra = {
            "example": {
                "query": "Giải thích khái niệm trí tuệ nhân tạo?",
                "model_name": "GPT-4",
                "model_type": "online",
                "api_key": "sk-xxxxxxxxx",  # Tùy chọn
                "endpoint": "https://api.openai.com/v1",  # Tùy chọn
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1024
                },
                "chat_section_id": "6488f9ae-13c7-42ff-b074-4850624bff5b"  # Tùy chọn
            }
        }


@router.post("/direct-llm/stream/")
async def direct_llm_chat_stream(request: DirectLLMRequest):
    """
    API streaming cho phép chat trực tiếp với mô hình LLM.
    Trả về Server-Sent Events (SSE) để streaming real-time.
    
    - Hỗ trợ đặc biệt cho Gemini streaming
    - Fallback to non-streaming nếu provider không hỗ trợ
    """
    from llms.stream_handlers import create_streaming_response, StreamingCallbackHandler
    import asyncio
    import json
    from fastapi.responses import StreamingResponse
    
    start_time = time.time()
    request_id = uuid.uuid4()

    logger.info(f"Request {request_id}: Processing STREAMING direct LLM chat with model {request.model_name} (type: {request.model_type})")
    
    async def generate_stream():
        try:
            # Lấy memory nếu có chat_section_id
            memory = None
            if request.chat_section_id:
                logger.info(f"[STREAMING_CHAT] Lấy memory cho chat_section_id: {request.chat_section_id}")
                memory = await QueryService.get_memory(request.chat_section_id)

            # Khởi tạo mô hình LLM
            model_type = request.model_type.lower()
            api_key = request.api_key

            if model_type == "online":
                llm = OnlineLLMs(model_name=request.model_name, api_key=api_key, endpoint=request.endpoint)
            else:  # local
                llm = LocalLlms(model_name=request.model_name, endpoint=request.endpoint)

            # Tạo prompt
            params = request.parameters or {}
            temperature = params.get("temperature", 0.7)
            max_tokens = params.get("max_tokens", 1024)

            if memory:
                custom_params = params.copy() if params else {}
                custom_params["conversation_memory"] = memory
                
                prompt_builder = {
                    "system_instruction": "Bạn là một trợ lý AI thông minh và hữu ích. Hãy sử dụng thông tin từ lịch sử cuộc trò chuyện để đưa ra câu trả lời phù hợp và có tính liên tục.",
                    "context_template": "",
                    "query_template": "Câu hỏi: {{query}}",
                    "instruction_template": "Hãy trả lời câu hỏi một cách chính xác, đầy đủ và có tính liên tục với cuộc trò chuyện trước đó."
                }
                
                # Sử dụng messages format tối ưu
                messages = PromptBuilderLoader.build_messages(prompt_builder, request.query, "", custom_params)
                prompt = PromptBuilderLoader._messages_to_prompt(messages)  # Convert về string cho compatibility
            else:
                prompt = f"Hãy trả lời câu hỏi sau một cách chính xác và đầy đủ: {request.query}"

            # Yield start signal
            yield f"data: {json.dumps({'token': '', 'started': True, 'request_id': str(request_id)})}\n\n"

            # Stream tokens from LLM
            full_response = ""
            async for token in llm.generate_content_stream(prompt, temperature=temperature, max_tokens=max_tokens):
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': token, 'finished': False})}\n\n"

            # Cập nhật memory nếu có chat_section_id - background task
            if request.chat_section_id and full_response:
                logger.info(f"[STREAMING_CHAT] Cập nhật memory cho chat_section_id: {request.chat_section_id}")
                
                llm_config = {
                    "model_type": request.model_type,
                    "model_name": request.model_name,
                    "api_key": api_key,
                    "endpoint": request.endpoint
                }
                
                # Background task
                asyncio.create_task(QueryService.update_memory(
                    request.chat_section_id,
                    request.query,
                    full_response,
                    memory,
                    llm=llm,
                    llm_config=llm_config
                ))

            # Yield completion signal with metadata
            end_time = time.time()
            processing_time = end_time - start_time
            
            metadata = {
                "request_id": str(request_id),
                "model": request.model_name,
                "processing_time": processing_time,
                "total_tokens": len(full_response.split()) if full_response else 0
            }
            
            yield f"data: {json.dumps({'token': '', 'finished': True, 'metadata': metadata})}\n\n"

        except Exception as e:
            logger.error(f"Request {request_id}: Error in streaming chat: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'token': '', 'error': str(e), 'finished': True})}\n\n"

    # Trả về StreamingResponse với SSE headers
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


@router.post("/direct-llm/", response_model=QueryResponse)
async def direct_llm_chat(request: DirectLLMRequest):
    """
    API cho phép chat trực tiếp với mô hình LLM mà không cần tìm kiếm thông tin từ knowledge base.
    
    - Nếu cung cấp chat_section_id, hệ thống sẽ sử dụng memory của cuộc hội thoại đó để cải thiện câu trả lời
    và tự động cập nhật memory sau mỗi lần tương tác.
    """
    start_time = time.time()
    request_id = uuid.uuid4()

    logger.info(f"Request {request_id}: Processing direct LLM chat with model {request.model_name} (type: {request.model_type})")
    
    # Log thông tin chat_section_id nếu có
    if request.chat_section_id:
        logger.info(f"Direct LLM chat with chat_section_id: {request.chat_section_id}")

    try:
        # Lấy memory nếu có chat_section_id
        memory = None
        if request.chat_section_id:
            logger.info(f"[DIRECT_CHAT_API] Lấy memory cho chat_section_id: {request.chat_section_id}")
            memory = await QueryService.get_memory(request.chat_section_id)
            if memory:
                logger.info(f"[DIRECT_CHAT_API] Đã tìm thấy memory cho chat_section_id: {request.chat_section_id}")
            else:
                logger.info(f"[DIRECT_CHAT_API] Chưa có memory cho chat_section_id: {request.chat_section_id}")

        # Sử dụng model_type được cung cấp trực tiếp từ request
        model_type = request.model_type.lower()
        
        # Kiểm tra API key nếu không được cung cấp trực tiếp trong request
        api_key = request.api_key

        # Khởi tạo mô hình LLM dựa trên loại model
        if model_type == "online":
            llm = OnlineLLMs(model_name=request.model_name, api_key=api_key, endpoint=request.endpoint)
        else:  # local
            llm = LocalLlms(model_name=request.model_name, endpoint=request.endpoint)

        # Lấy tham số LLM từ request (nếu có)
        params = request.parameters or {}
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 1024)

        # Tạo prompt, kết hợp với memory nếu có
        if memory:
            # Tạo bản sao của params để tránh thay đổi cấu trúc gốc
            custom_params = params.copy() if params else {}
            
            # Thêm memory vào parameters
            custom_params["conversation_memory"] = memory
            logger.info(f"[DIRECT_CHAT_API] Đã bổ sung memory vào prompt")
            
            # Tạo prompt đơn giản với memory
            prompt_builder = {
                "system_instruction": "Bạn là một trợ lý AI thông minh và hữu ích. Hãy sử dụng thông tin từ lịch sử cuộc trò chuyện để đưa ra câu trả lời phù hợp và có tính liên tục.",
                "context_template": "",  # Không có context trong direct chat
                "query_template": "Câu hỏi: {{query}}",
                "instruction_template": "Hãy trả lời câu hỏi một cách chính xác, đầy đủ và có tính liên tục với cuộc trò chuyện trước đó."
            }
            
            # Tạo prompt với memory
            logger.info(f"[DIRECT_CHAT_API] Xây dựng prompt có tích hợp memory")
            # Sử dụng messages format tối ưu
            messages = PromptBuilderLoader.build_messages(prompt_builder, request.query, "", custom_params)
            prompt = PromptBuilderLoader._messages_to_prompt(messages)  # Convert về string cho compatibility
        else:
            # Nếu không có memory, sử dụng prompt đơn giản
            prompt = f"Hãy trả lời câu hỏi sau một cách chính xác và đầy đủ: {request.query}"

        # Gọi LLM để tạo câu trả lời
        llm_response = llm.generate_content_with_timing(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        answer = llm_response["content"]

        # Cập nhật memory nếu có chat_section_id
        if request.chat_section_id:
            logger.info(f"[DIRECT_CHAT_API] Cập nhật memory cho chat_section_id: {request.chat_section_id}")
            
            # Tạo cấu hình LLM để cập nhật memory
            llm_config = {
                "model_type": request.model_type,
                "model_name": request.model_name,
                "api_key": api_key,
                "endpoint": request.endpoint
            }
            
            # Gọi hàm update_memory của QueryService
            await QueryService.update_memory(
                request.chat_section_id,
                request.query,
                answer,
                memory,
                llm=llm,
                llm_config=llm_config
            )

        # Tính thời gian xử lý
        end_time = time.time()
        processing_time = end_time - start_time

        # Log kết quả
        logger.info(f"Request {request_id}: Direct LLM chat processed in {processing_time:.2f}s")

        # Trả về cấu trúc giống với QueryResponse nhưng chỉ có một số trường có giá trị
        return {
            "query": request.query,
            "retrieved_chunks": [],  # Danh sách rỗng cho retrieved_chunks
            "llm_answer": answer,
            "model": request.model_name,
            "config_name": f"Direct LLM ({request.model_type})",  # Thêm thông tin model_type
            "config_id": uuid.uuid4(),  # UUID dummy
            "request_id": request_id,
            "processing_time": processing_time
        }

    except Exception as e:
        end_time = time.time()
        logger.error(f"Request {request_id}: Error processing direct LLM chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý direct LLM chat: {str(e)}")


# Endpoint để kiểm tra các nhà cung cấp AI hỗ trợ
@router.get("/direct-llm/providers")
async def get_providers():
    """Lấy danh sách các nhà cung cấp AI hỗ trợ"""
    from llms.onlinellms import PROVIDER_HINTS
    providers = list(PROVIDER_HINTS.keys())
    
    return {
        "supported_providers": providers,
        "provider_examples": {
            "openai": ["gpt-4-turbo", "gpt-3.5-turbo", "text-embedding-3-small"],
            "gemini": ["gemini-pro"],
            "cohere": ["command-r", "command-light"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet"],
            "mistral": ["mistral-large", "mistral-medium"],
        }
    }


# Endpoint để kiểm tra tình trạng API key
@router.get("/direct-llm/check-api-keys")
async def check_api_keys():
    """Kiểm tra tính khả dụng của các API key từ file .env"""
    result = {
        "openai": os.getenv("OPENAI_API_KEY") is not None,
        "gemini": os.getenv("GEMINI_API_KEY") is not None,
        "cohere": os.getenv("COHERE_API_KEY") is not None,
        "anthropic": os.getenv("ANTHROPIC_API_KEY") is not None,
        "mistral": os.getenv("MISTRAL_API_KEY") is not None,
        "local_models_available": True  # Giả định mô hình local luôn khả dụng
    }

    return result