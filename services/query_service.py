import uuid
import time
import logging
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from database.db_connection import get_db_pool, get_pg_connection, return_pg_connection
from pgvector.psycopg2 import register_vector
from database.query_config_db import QueryConfigDB
from llms.config_loader import ModelConfigLoader, PromptBuilderLoader
from models.config_models import QueryRequest, ChunkResponse
import backoff
from api.chunks_embedding_api import pad_embedding_vector, embedding_config
from datetime import datetime
from services.keyword_extractor import extract_keywords
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from functools import lru_cache
from services.query_cache import query_cache
from services.rerank_cache import rerank_cache

logger = logging.getLogger(__name__)

# Khởi tạo ThreadPoolExecutor một cách toàn cục
# os.cpu_count() có thể trả về None, nên cần fallback an toàn
MAX_WORKERS = max(os.cpu_count() * 3, 12) if os.cpu_count() else 12
thread_pool_executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS,
    thread_name_prefix="QueryService"
)

class GlobalStateManager:
    """
    Quản lý trạng thái toàn cục của hệ thống, bao gồm các tài nguyên dùng chung.
    Sử dụng singleton pattern để đảm bảo tài nguyên được sử dụng hiệu quả.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalStateManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Khởi tạo các tài nguyên và trạng thái ban đầu"""
        self._reranker_models = {}
        self._loading_locks = {}
        self._loading_states = {}
        logger.info("[GLOBAL_STATE] Khởi tạo GlobalStateManager")
    
    async def get_reranker(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Lấy reranker model theo tên, đảm bảo không bị tải lại nếu đã tồn tại.
        Tối ưu hóa: Mỗi model chỉ được tải một lần trong suốt vòng đời ứng dụng.
        Hoặc sử dụng Model Server nếu được cấu hình.
        """
        # Kiểm tra cơ chế Model Server
        use_model_server = os.getenv('USE_MODEL_SERVER', 'false').lower() == 'true'
        worker_id = os.getenv('WORKER_ID', '1')
        
        if use_model_server and worker_id != '1':
            # Workers khác sử dụng Model Client
            from services.model_server import get_model_client
            model_client = get_model_client()
            if model_client.health_check():
                logger.info(f"[GLOBAL_STATE] Worker {worker_id}: Sử dụng Model Server cho reranking")
                return "model_server", model_client
            else:
                logger.warning(f"[GLOBAL_STATE] Worker {worker_id}: Model Server không khả dụng, fallback về local")
        # Tạo lock cho model này nếu chưa tồn tại
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = asyncio.Lock()
            self._loading_states[model_name] = False
            
        # Nếu đã tải xong, trả về kết quả ngay
        if model_name in self._reranker_models:
            return self._reranker_models[model_name]
            
        # Nếu đang tải, đợi cho đến khi hoàn tất
        async with self._loading_locks[model_name]:
            if self._loading_states[model_name]:
                # Nếu đã có tiến trình khác đang tải, chúng ta chờ và kiểm tra lại
                # Điều này đảm bảo không có hai tiến trình cùng tải một model
                while self._loading_states[model_name]:
                    # Đợi một chút trước khi kiểm tra lại
                    await asyncio.sleep(0.1)
                
                # Kiểm tra xem model đã tải xong chưa sau khi đợi
                if model_name in self._reranker_models:
                    return self._reranker_models[model_name]
            
            # Đánh dấu là đang tải
            self._loading_states[model_name] = True
            
            try:
                # Tải model trong thread riêng
                loop = asyncio.get_event_loop()
                logger.info(f"[GLOBAL_STATE] Tải reranker model: {model_name}")
                tokenizer, model = await loop.run_in_executor(
                    thread_pool_executor,
                    lambda: ModelConfigLoader.load_reranker_sync(model_name)
                )
                
                # Lưu kết quả vào cache
                if tokenizer is not None and model is not None:
                    self._reranker_models[model_name] = (tokenizer, model)
                    logger.info(f"[GLOBAL_STATE] Đã tải xong reranker model: {model_name}")
                    return self._reranker_models[model_name]
                else:
                    logger.error(f"[GLOBAL_STATE] Không thể tải reranker model: {model_name}")
                    return None, None
                    
            except Exception as e:
                logger.error(f"[GLOBAL_STATE] Lỗi khi tải reranker model: {str(e)}", exc_info=True)
                return None, None
                
            finally:
                # Đánh dấu là đã hoàn tất việc tải
                self._loading_states[model_name] = False

# Khởi tạo instance duy nhất của GlobalStateManager
global_state = GlobalStateManager()

class RerankerSingleton:
    _instance = None
    _tokenizer = None
    _model = None
    _last_used_model_name = None
    _loading = False
    _loading_lock = asyncio.Lock()
    
    @classmethod
    async def get_instance(cls, model_name: str = "BAAI/bge-reranker-base"):
        """
        Trả về instance của reranker theo kiểu singleton.
        Sử dụng GlobalStateManager để quản lý model.
        """
        # Sử dụng GlobalStateManager để lấy model
        tokenizer, model = await global_state.get_reranker(model_name)
        
        if tokenizer is not None and model is not None:
            cls._tokenizer = tokenizer
            cls._model = model
            cls._last_used_model_name = model_name
            cls._instance = cls
            return cls._instance
        else:
            logger.error(f"[RERANKER] Không thể lấy reranker model: {model_name} từ GlobalStateManager")
            return None
    
    @classmethod
    def get_tokenizer_and_model(cls):
        """Trả về tokenizer và model của reranker instance"""
        return cls._tokenizer, cls._model

class QueryService:
    @staticmethod
    async def _generate_embedding_with_retry(text: str, llm_config: dict, max_retries: int = 3) -> List[float]:
        """
        Tạo embedding với retry mechanism khi gặp lỗi rate limit.
        Sẽ thử với các API key khác nhau nếu gặp lỗi 429.
        """
        from llms.config_loader import mark_rate_limited
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[QUERY_SERVICE] Thử tạo embedding lần {attempt + 1}/{max_retries}")
                embedding_model = ModelConfigLoader.load_embedding_model(llm_config)
                return embedding_model.generate_embedding(text)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['429', 'quota', 'rate limit', 'exceeded']):
                    logger.warning(f"[QUERY_SERVICE] Lỗi rate limit lần {attempt + 1}: {str(e)}")
                    
                    # Đánh dấu API key hiện tại bị rate limit
                    if hasattr(embedding_model, 'api_key') and embedding_model.api_key:
                        mark_rate_limited(embedding_model.api_key, seconds=900)
                        logger.info(f"[QUERY_SERVICE] Đã đánh dấu API key bị rate limit, sẽ thử với API key khác")
                    
                    # Nếu chưa hết số lần thử, tiếp tục với API key khác
                    if attempt < max_retries - 1:
                        logger.info(f"[QUERY_SERVICE] Thử lại với API key khác...")
                        continue
                    else:
                        logger.error(f"[QUERY_SERVICE] Đã thử hết {max_retries} lần, tất cả API key đều bị rate limit")
                        raise
                else:
                    # Lỗi khác không phải rate limit, throw ngay
                    logger.error(f"[QUERY_SERVICE] Lỗi không phải rate limit: {str(e)}")
                    raise
        
        # Fallback nếu không thành công
        raise Exception(f"Không thể tạo embedding sau {max_retries} lần thử")

    @staticmethod
    async def _generate_content_with_retry(llm, prompt: str, temperature: float = 0.7, max_tokens: int = 1024, max_retries: int = 3):
        """
        Tạo content với retry mechanism khi gặp lỗi rate limit.
        Sẽ thử với các API key khác nhau nếu gặp lỗi 429.
        """
        from llms.config_loader import mark_rate_limited
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[QUERY_SERVICE] Thử tạo content lần {attempt + 1}/{max_retries}")
                return llm.generate_content(prompt, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['429', 'quota', 'rate limit', 'exceeded']):
                    logger.warning(f"[QUERY_SERVICE] Lỗi rate limit lần {attempt + 1}: {str(e)}")
                    
                    # Đánh dấu API key hiện tại bị rate limit
                    if hasattr(llm, 'api_key') and llm.api_key:
                        mark_rate_limited(llm.api_key, seconds=900)
                        logger.info(f"[QUERY_SERVICE] Đã đánh dấu API key bị rate limit, sẽ thử với API key khác")
                    
                    # Nếu chưa hết số lần thử, tiếp tục với API key khác
                    if attempt < max_retries - 1:
                        logger.info(f"[QUERY_SERVICE] Thử lại với API key khác...")
                        # Tạo LLM instance mới với API key khác
                        from llms.config_loader import ModelConfigLoader
                        # Cần lấy config từ đâu đó - tạm thời skip retry cho LLM
                        logger.warning(f"[QUERY_SERVICE] Không thể retry LLM với API key khác vì thiếu config")
                        raise
                    else:
                        logger.error(f"[QUERY_SERVICE] Đã thử hết {max_retries} lần, tất cả API key đều bị rate limit")
                        raise
                else:
                    # Lỗi khác không phải rate limit, throw ngay
                    logger.error(f"[QUERY_SERVICE] Lỗi không phải rate limit: {str(e)}")
                    raise
        
        # Fallback nếu không thành công
        raise Exception(f"Không thể tạo content sau {max_retries} lần thử")

    @staticmethod
    async def classify_query(query: str, llm) -> bool:
        """
        Phân loại câu hỏi thành 2 nhóm:
        - True: Chào hỏi, hỏi thăm, cảm ơn, ngạc nhiên, cảm xúc (không cần RAG)
        - False: Câu hỏi về kiến thức (cần RAG)
        
        Args:
            query: Câu hỏi cần phân loại
            llm: Model LLM đã được khởi tạo
            
        Returns:
            True nếu là câu chào hỏi, cảm ơn, hỏi thăm, False nếu là câu hỏi kiến thức
        """
        prompt = f"""
        Hãy phân loại câu hỏi sau đây vào một trong hai nhóm:

        Câu hỏi: "{query}"

        Nhóm 1: Chào hỏi, hỏi thăm, cảm ơn, ngạc nhiên, thể hiện cảm xúc, trêu đùa, hời hợt
        Nhóm 2: Câu hỏi về kiến thức, yêu cầu tìm kiếm, kiểm tra thông tin liên quan đến điểm số, học tập, quy định quy chế đào tạo, giờ học, tín chỉ

        Chỉ trả lời "Nhóm 1" hoặc "Nhóm 2", không giải thích thêm.
        """
        
        try:
            start_time = time.time()
            logger.info(f"[QUERY_SERVICE] Phân loại câu hỏi: {query}")
            
            response = llm.generate_content(prompt)
            
            # Xử lý cả trường hợp response là dict hoặc string
            content = ""
            if isinstance(response, dict) and "content" in response:
                content = response["content"]
            elif isinstance(response, str):
                content = response
                
            content = content.strip().lower()
            
            # Tính thời gian phân loại
            end_time = time.time()
            classification_time = end_time - start_time
            
            is_conversational = "nhóm 1" in content or "group 1" in content
            
            logger.info(f"[QUERY_SERVICE] Kết quả phân loại: {'Nhóm 1 (hội thoại)' if is_conversational else 'Nhóm 2 (kiến thức)'}")
            logger.info(f"[QUERY_SERVICE] Thời gian phân loại: {classification_time:.2f}s")
            
            return is_conversational
            
        except Exception as e:
            logger.error(f"[QUERY_SERVICE] Lỗi khi phân loại câu hỏi: {str(e)}", exc_info=True)
            # Mặc định là câu hỏi kiến thức nếu có lỗi
            return False

    async def _enhance_query_for_search(self, original_query: str, memory: Optional[Dict[str, Any]], llm_instance: Any) -> str:
        """
        Cải thiện câu hỏi cho mục đích tìm kiếm bằng cách sử dụng LLM và memory.
        Trả về câu hỏi đã cải thiện hoặc câu hỏi gốc nếu không cần hoặc không thể cải thiện.
        """
        if not llm_instance:
            logger.warning("[QUERY_SERVICE] No LLM instance provided for query enhancement. Returning original query.")
            return original_query

        context_from_memory = ""
        if memory:
            try:
                last_questions_list = memory.get("last_questions", [])
                if last_questions_list and isinstance(last_questions_list, list):
                    # Lấy tối đa 2 câu hỏi gần nhất, xử lý cả trường hợp có dict và str
                    recent_qs = []
                    for q in last_questions_list[-2:]:
                        if isinstance(q, dict) and q.get("question"):
                            recent_qs.append(q.get("question"))
                        elif isinstance(q, str):
                            recent_qs.append(q)
                            
                    if recent_qs:
                        context_from_memory += "Các câu hỏi trước đó:\n" + "\n".join([f"- {q}" for q in recent_qs])
                
                summary = memory.get("summary", "")
                if summary:
                    context_from_memory += "\nTóm tắt hội thoại:\n" + summary
            except Exception as e:
                logger.error(f"[QUERY_SERVICE] Error building context from memory for enhancement: {e}", exc_info=True)
        
        prompt = f"""
Bạn là một trợ lý AI có nhiệm vụ làm rõ câu hỏi của người dùng để tối ưu hóa việc tìm kiếm thông tin.
Dựa vào câu hỏi gốc và ngữ cảnh hội thoại (nếu có), hãy quyết định:

1.  **Nếu câu hỏi gốc đã rất rõ ràng, đầy đủ, và có thể hiểu độc lập** (ví dụ: "Học phí ngành Công nghệ thông tin năm 2024 là bao nhiêu?", "Xin chào"): Hãy trả lại CHÍNH XÁC câu hỏi gốc.
2.  **Nếu câu hỏi gốc ngắn, mơ hồ, hoặc phụ thuộc vào ngữ cảnh** (ví dụ: "còn cái đó thì sao?", "kể tiếp đi", "đúng vậy không?", "thế nào?"): Hãy viết lại thành một câu hỏi đầy đủ, rõ ràng hơn bằng cách kết hợp thông tin từ câu hỏi gốc và ngữ cảnh. Giữ nguyên ý định chính.

Ngữ cảnh hội thoại (nếu có):
---
{context_from_memory if context_from_memory else "Không có ngữ cảnh."}
---

Câu hỏi gốc của người dùng: "{original_query}"

Câu hỏi làm rõ (hoặc câu hỏi gốc nếu không cần thay đổi):
        """

        try:
            logger.info(f"[QUERY_SERVICE] Enhancing query for search: '{original_query}'")
            response_data = llm_instance.generate_content(prompt)
            
            enhanced_q = original_query # Mặc định
            if isinstance(response_data, dict) and "content" in response_data:
                enhanced_q = response_data["content"].strip()
            elif isinstance(response_data, str):
                enhanced_q = response_data.strip()
            
            if not enhanced_q: # Nếu LLM trả về rỗng, dùng câu gốc
                enhanced_q = original_query

            logger.info(f"[QUERY_SERVICE] Original query: '{original_query}' -> Enhanced for search: '{enhanced_q}'")
            return enhanced_q
        except Exception as e:
            logger.error(f"[QUERY_SERVICE] Error enhancing query for search: {e}. Returning original query.", exc_info=True)
            return original_query

    @staticmethod
    async def execute_query_stream(request: QueryRequest, chat_section_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Thực hiện truy vấn RAG với streaming response.
        
        Args:
            request: Đối tượng QueryRequest chứa thông tin truy vấn
            chat_section_id: ID của chat section để quản lý memory (optional)
            
        Returns:
            Dict[str, Any]: Kết quả truy vấn với streaming response
            
        Raises:
            HTTPException: Khi có lỗi trong quá trình xử lý
            ValueError: Khi request không hợp lệ
        """
        import json
        
        start_time = time.time()
        request_id = uuid.uuid4()
        
        logger.info(f"Request {request_id}: Processing STREAMING RAG query: {request.query}")
        
        try:
            # Load configuration (same as regular RAG)
            config = None
            if request.config_id:
                config = await QueryConfigDB.get_config(request.config_id)
            else:
                # Get default config for knowledge base
                config = await QueryConfigDB.get_default_config(request.knowledgeBaseId)
                if not config:
                    # Fallback: get any config for this knowledge base
                    configs = await QueryConfigDB.get_configs_by_knowledge_base(request.knowledgeBaseId)
                    if configs:
                        config = configs[0]

            if not config:
                yield f"data: {json.dumps({'error': 'Không tìm thấy cấu hình phù hợp', 'finished': True})}\n\n"
                return

            # Get memory if chat_section_id provided
            memory = None
            if chat_section_id:
                memory = await QueryService.get_memory(chat_section_id)

            # Load models (same as regular RAG)
            llm_config = config["llm_config"]
            llm = ModelConfigLoader.load_model(llm_config)
            
            # Yield start signal with config info
            yield f"data: {json.dumps({'started': True, 'request_id': str(request_id), 'config_name': config.get('name_config'), 'model': llm_config.get('model_name')})}\n\n"

            # Query enhancement and classification (same as regular RAG)
            query_service_instance = QueryService()
            enhanced_query_for_search = await query_service_instance._enhance_query_for_search(request.query, memory, llm)
            is_conversational = await QueryService.classify_query(enhanced_query_for_search, llm)
            
            # Direct response for conversational queries
            if is_conversational:
                logger.info(f"[STREAMING_RAG] Conversational query, direct response")
                
                yield f"data: {json.dumps({'status': 'conversational_query', 'chunks': []})}\n\n"
                
                # Build prompt for direct answer (same logic as regular RAG)
                if memory:
                    custom_params = request.parameters.copy() if request.parameters else {}
                    custom_params["conversation_memory"] = memory
                else:
                    custom_params = request.parameters
                
                kb_info = None
                if config["prompt_builder"].get("parameters", {}).get("include_kb_description", True):
                    kb_info = await QueryService.get_knowledge_base_info(request.knowledgeBaseId)
                
                # Sử dụng messages format tối ưu với conversation history
                messages_for_direct_answer = PromptBuilderLoader.build_messages(
                    config["prompt_builder"], request.query, "", custom_params, kb_info=kb_info, chat_section_id=request.chat_section_id
                )
                
                llm_params = PromptBuilderLoader.get_parameters(config["prompt_builder"], request.parameters)
                temperature = llm_params.get("temperature", 0.7)
                max_tokens = llm_params.get("max_tokens", 1024)
                
                # Stream direct answer với messages format
                full_response = ""
                async for token in llm.generate_content_stream_with_messages(messages_for_direct_answer, temperature=temperature, max_tokens=max_tokens):
                    if token:
                        full_response += token
                        yield f"data: {json.dumps({'token': token, 'finished': False})}\n\n"
                
                # Background memory update và conversation history
                if chat_section_id:
                    asyncio.create_task(QueryService.update_memory_background(
                        chat_section_id, request.query, full_response, memory,
                        llm=llm, llm_config=llm_config
                    ))
                    
                    # Lưu conversation history
                    asyncio.create_task(QueryService._save_conversation_history(
                        chat_section_id, request.query, full_response
                    ))
                
                # End signal
                end_time = time.time()
                processing_time = end_time - start_time
                metadata = {
                    "request_id": str(request_id),
                    "processing_time": processing_time,
                    "total_tokens": len(full_response.split()) if full_response else 0,
                    "type": "direct_response"
                }
                yield f"data: {json.dumps({'token': '', 'finished': True, 'metadata': metadata})}\n\n"
                return
            
            # Knowledge-based RAG flow (streaming version)
            logger.info(f"[STREAMING_RAG] Knowledge query, performing RAG")
            
            yield f"data: {json.dumps({'status': 'rag_processing', 'stage': 'embedding'})}\n\n"
            
            # Load embedding model and create query embedding with retry mechanism
            query_embedding = await QueryService._generate_embedding_with_retry(enhanced_query_for_search, llm_config)
            
            # Pad embedding vector
            from api.chunks_embedding_api import pad_embedding_vector, embedding_config
            query_embedding = pad_embedding_vector(
                query_embedding, 
                target_dim=embedding_config.target_dimension,
                method=embedding_config.embedding_method
            )
            
            yield f"data: {json.dumps({'status': 'rag_processing', 'stage': 'search'})}\n\n"
            
            # Extract keywords and search chunks (same as regular RAG)
            query_keywords = await extract_keywords(enhanced_query_for_search, method="nlp", llm_instance=llm)
            
            # Search parameters
            params = config["prompt_builder"].get("parameters", {})
            max_chunks = params.get("max_chunks", 10)
            use_reranker = params.get("use_reranker", True)
            
            # Load reranker
            reranker_model_name = llm_config.get("reranker_model", "BAAI/bge-reranker-base")
            reranker_instance = await RerankerSingleton.get_instance(reranker_model_name)
            reranker_tokenizer, reranker_model = None, None
            if reranker_instance:
                reranker_tokenizer, reranker_model = reranker_instance.get_tokenizer_and_model()
            
            # Search related chunks (same as regular RAG)
            related_chunks = await QueryService.search_related_chunks(
                query_embedding, enhanced_query_for_search, query_keywords,
                request.knowledgeBaseId, max_chunks=max_chunks,
                use_reranker=use_reranker, reranker_tokenizer=reranker_tokenizer,
                reranker_model=reranker_model, reranker_model_name=reranker_model_name
            )
            
            # Prepare chunks info for streaming
            chunks_info = []
            for chunk in related_chunks:
                chunks_info.append({
                    "chunk_id": str(chunk.chunk_id),
                    "document_name": chunk.document_name,
                    "similarity_score": chunk.similarity_score,
                    "rerank_score": chunk.rerank_score
                })
            
            yield f"data: {json.dumps({'status': 'rag_processing', 'stage': 'chunks_found', 'chunks_count': len(related_chunks)})}\n\n"
            yield f"data: {json.dumps({'status': 'chunks_info', 'chunks': chunks_info})}\n\n"
            
            # Build context from chunks (same as regular RAG)
            context_parts = []
            for chunk in related_chunks:
                # Đảm bảo document_name không rỗng trước khi thêm vào
                doc_name_prefix = f"[Nguồn: {chunk.document_name}] " if chunk.document_name else ""
                context_parts.append(f"{doc_name_prefix}{chunk.chunk_text}")
            context = "\n\n".join(context_parts)
            logger.info(f"[STREAMING_RAG] Xây dựng context với {len(context)} ký tự từ {len(related_chunks)} chunks")
            
            # Build final prompt with memory
            if memory:
                custom_params = request.parameters.copy() if request.parameters else {}
                custom_params["conversation_memory"] = memory
            else:
                custom_params = request.parameters
            
            kb_info = None
            if config["prompt_builder"].get("parameters", {}).get("include_kb_description", True):
                kb_info = await QueryService.get_knowledge_base_info(request.knowledgeBaseId)
            
            # Sử dụng messages format tối ưu cho RAG với conversation history
            messages_for_answer = PromptBuilderLoader.build_messages(
                config["prompt_builder"], request.query, context, custom_params, kb_info=kb_info, chat_section_id=request.chat_section_id
            )
            
            yield f"data: {json.dumps({'status': 'rag_processing', 'stage': 'llm_generating'})}\n\n"
            
            # Stream LLM response với messages format (THE KEY DIFFERENCE)
            llm_params = PromptBuilderLoader.get_parameters(config["prompt_builder"], request.parameters)
            temperature = llm_params.get("temperature", 0.7)
            max_tokens = llm_params.get("max_tokens", 1024)
            
            full_response = ""
            async for token in llm.generate_content_stream_with_messages(messages_for_answer, temperature=temperature, max_tokens=max_tokens):
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': token, 'finished': False})}\n\n"
            
            # Background memory update và conversation history
            if chat_section_id:
                asyncio.create_task(QueryService.update_memory_background(
                    chat_section_id, request.query, full_response, memory,
                    llm=llm, llm_config=llm_config
                ))
                
                # Lưu conversation history
                asyncio.create_task(QueryService._save_conversation_history(
                    chat_section_id, request.query, full_response
                ))
            
            # Final metadata
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Serialize chunks for final response
            serializable_chunks = []
            for chunk in related_chunks:
                serializable_chunks.append({
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "document_link": chunk.document_link,
                    "chunk_text": chunk.chunk_text,
                    "document_name": chunk.document_name,
                    "similarity_score": chunk.similarity_score,
                    "rerank_score": chunk.rerank_score
                })
            
            metadata = {
                "request_id": str(request_id),
                "processing_time": processing_time,
                "total_tokens": len(full_response.split()) if full_response else 0,
                "chunks_count": len(related_chunks),
                "retrieved_chunks": serializable_chunks,
                "type": "rag_response",
                "model": llm_config.get("model_name"),
                "config_name": config["name_config"]
            }
            
            yield f"data: {json.dumps({'token': '', 'finished': True, 'metadata': metadata})}\n\n"
            
        except Exception as e:
            logger.error(f"Request {request_id}: Error in streaming RAG: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'token': '', 'error': str(e), 'finished': True})}\n\n"

    @staticmethod
    async def execute_query(request: QueryRequest, chat_section_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Thực hiện truy vấn RAG hoàn chỉnh với cache và memory management.
        
        Args:
            request: Đối tượng QueryRequest chứa thông tin truy vấn
            chat_section_id: ID của chat section để quản lý memory (optional)
            
        Returns:
            Dict[str, Any]: Kết quả truy vấn bao gồm:
                - answer: Câu trả lời từ LLM
                - chunks: Danh sách chunks được sử dụng
                - processing_time: Thời gian xử lý
                - request_id: ID của request
                
        Raises:
            HTTPException: Khi có lỗi trong quá trình xử lý
            ValueError: Khi request không hợp lệ
        """
        start_time = time.time()
        request_id = uuid.uuid4()

        logger.info(f"Request {request_id}: Processing query for KB {request.knowledgeBaseId}")

        # 🎯 CACHE CHECK - Kiểm tra cache trước khi xử lý
        config_id = str(request.config_id) if request.config_id else "default"
        cached_result = query_cache.get(request.query, str(request.knowledgeBaseId), config_id)
        if cached_result:
            # Cache hit - trả về kết quả ngay lập tức
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Tạo copy để không modify cache gốc
            result = cached_result.copy()
            result["request_id"] = str(request_id)
            result["processing_time"] = processing_time
            result["from_cache"] = True
            
            logger.info(f"🎯 Request {request_id}: Cache HIT! Served in {processing_time:.3f}s")
            return result

        try:
            # Lấy memory nếu có chat_section_id
            memory = None
            if chat_section_id:
                logger.info(f"[QUERY_SERVICE] Lấy memory cho chat_section_id: {chat_section_id}")
                memory = await QueryService.get_memory(chat_section_id)
                if memory:
                    logger.info(f"[QUERY_SERVICE] Đã tìm thấy memory cho chat_section_id: {chat_section_id}")
                else:
                    logger.info(f"[QUERY_SERVICE] Chưa có memory cho chat_section_id: {chat_section_id}")

            # Lấy cấu hình từ database
            config = None
            if request.config_id:
                logger.info(f"[QUERY_SERVICE] Lấy config từ DB với config_id: {request.config_id}")
                config = await QueryConfigDB.get_config(request.config_id)
                if config:
                    logger.info(f"[QUERY_SERVICE] Đã tìm thấy config với ID: {request.config_id}")
                else:
                    logger.warning(f"[QUERY_SERVICE] Không tìm thấy config với ID: {request.config_id}")

            if not config:
                # Sử dụng cấu hình mặc định nếu không có config_id hoặc config_id không hợp lệ
                logger.info(f"[QUERY_SERVICE] Lấy config mặc định cho knowledge base: {request.knowledgeBaseId}")
                config = await QueryConfigDB.get_default_config(request.knowledgeBaseId)
                if config:
                    logger.info(f"[QUERY_SERVICE] Đã tìm thấy config mặc định cho KB: {request.knowledgeBaseId}")
                else:
                    logger.warning(f"[QUERY_SERVICE] Không tìm thấy config mặc định cho KB: {request.knowledgeBaseId}")

            if not config:
                logger.warning(f"Request {request_id}: No config found for KB {request.knowledgeBaseId}")
                raise ValueError("Không tìm thấy cấu hình phù hợp")

            # Log thông tin config đã tìm thấy
            config_id = config.get("id")
            config_name = config.get("name_config")
            logger.info(f"[QUERY_SERVICE] Sử dụng config: {config_name} (ID: {config_id})")
            
            # Tạo bản sao của llm_config để log (che API key)
            llm_config_log = config.get("llm_config", {}).copy()
            if "api_key" in llm_config_log and llm_config_log["api_key"]:
                api_key = llm_config_log["api_key"]
                llm_config_log["api_key"] = f"{api_key[:3]}...{api_key[-3:]}" if len(api_key) > 6 else "***"
            
            logger.info(f"[QUERY_SERVICE] LLM Config: {json.dumps(llm_config_log, indent=2, ensure_ascii=False)}")
            prompt_log = {k: v for k, v in config.get("prompt_builder", {}).items() if k != "system_instruction"}
            logger.info(f"[QUERY_SERVICE] Prompt Builder Config: {json.dumps(prompt_log, indent=2, ensure_ascii=False)}")

            # Log các tham số tùy chỉnh từ request nếu có
            if request.parameters:
                logger.info(f"[QUERY_SERVICE] Tham số tùy chỉnh từ request: {json.dumps(request.parameters, indent=2, ensure_ascii=False)}")
                if "custom_system_instruction" in request.parameters:
                    custom_system = request.parameters.get("custom_system_instruction", "")
                    logger.info(f"[QUERY_SERVICE] Nhận custom_system_instruction: {custom_system[:50]}...")
                if "custom_instruction_template" in request.parameters:
                    custom_instruction = request.parameters.get("custom_instruction_template", "")
                    logger.info(f"[QUERY_SERVICE] Nhận custom_instruction_template: {custom_instruction[:50]}...")

            # Tải mô hình
            llm_config = config["llm_config"]
            logger.info(f"[QUERY_SERVICE] Tải LLM model từ config")
            llm = ModelConfigLoader.load_model(llm_config)
            
            # Tạo instance của QueryService để gọi _enhance_query_for_search
            query_service_instance = QueryService()

            # Cải thiện câu hỏi cho tìm kiếm (sử dụng LLM chính)
            enhanced_query_for_search = await query_service_instance._enhance_query_for_search(request.query, memory, llm)
            
            # Phân loại câu hỏi bằng LLM (sử dụng câu hỏi đã cải thiện nếu có)
            is_conversational = await QueryService.classify_query(enhanced_query_for_search, llm)
            
            # Nếu là câu hỏi hội thoại (chào hỏi, cảm ơn, cảm xúc...), trả lời trực tiếp mà không cần RAG
            if is_conversational:
                logger.info(f"[QUERY_SERVICE] Câu hỏi được phân loại là hội thoại, trả lời trực tiếp mà không cần RAG")
                
                # Tạo prompt đơn giản với memory nếu có
                if memory:
                    # Tạo bản sao của params để tránh thay đổi cấu trúc gốc
                    custom_params = request.parameters.copy() if request.parameters else {}
                    
                    # Thêm memory vào parameters để truyền đến build_prompt
                    custom_params["conversation_memory"] = memory
                    logger.info(f"[QUERY_SERVICE] Đã bổ sung memory vào prompt")
                else:
                    custom_params = request.parameters
                
                # Lấy thông tin knowledge base cho trường hợp trò chuyện trực tiếp
                kb_info = None
                if config["prompt_builder"].get("parameters", {}).get("include_kb_description", True):
                    kb_info = await QueryService.get_knowledge_base_info(request.knowledgeBaseId)
                    if kb_info:
                        logger.info(f"[QUERY_SERVICE] Đã lấy thông tin knowledge base cho trò chuyện trực tiếp: {kb_info.get('title', '')}")
                
                # Sử dụng prompt builder nhưng bỏ qua phần context
                # Dùng câu hỏi GỐC để build prompt cho LLM trả lời trực tiếp
                prompt_for_direct_answer = PromptBuilderLoader.build_prompt(
                    config["prompt_builder"], 
                    request.query, 
                    "", 
                    custom_params,
                    kb_info=kb_info
                )
                
                # Lấy tham số cho LLM
                llm_params = PromptBuilderLoader.get_parameters(config["prompt_builder"], request.parameters)
                temperature = llm_params.get("temperature", 0.7)
                max_tokens = llm_params.get("max_tokens", 1024)
                
                # Gọi LLM để tạo câu trả lời
                max_tokens = int(max_tokens) if max_tokens is not None else 1024
                logger.info(f"[QUERY_SERVICE] Gọi LLM trực tiếp với temperature={temperature}, max_tokens={max_tokens}")
                llm_response = llm.generate_content_with_timing(prompt_for_direct_answer, temperature=temperature, max_tokens=max_tokens)
                llm_answer = llm_response["content"]
                llm_time = llm_response.get("processing_time", 0)
                logger.info(f"[QUERY_SERVICE] LLM trả về câu trả lời với {len(llm_answer)} ký tự trong {llm_time:.2f}s")
                
                # Cập nhật memory nếu có chat_section_id - BACKGROUND TASK
                if chat_section_id:
                    logger.info(f"[QUERY_SERVICE] Khởi tạo background memory update cho chat_section_id: {chat_section_id}")
                    # Background task - không chờ
                    asyncio.create_task(QueryService.update_memory_background(
                        chat_section_id, 
                        request.query, 
                        llm_answer, 
                        memory,
                        llm=llm,
                        llm_config=llm_config
                    ))
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                logger.info(f"Request {request_id}: Query processed (direct response) in {processing_time:.2f}s")
                
                # Tạo response cho direct answer
                response = {
                    "query": request.query,
                    "retrieved_chunks": [],  # Không có chunks vì không qua RAG
                    "llm_answer": llm_answer,
                    "model": llm_config.get("model_name"),
                    "config_name": config["name_config"],
                    "config_id": str(config["id"]),  # Convert UUID to string
                    "request_id": str(request_id),   # Convert UUID to string
                    "processing_time": processing_time,
                    "from_cache": False
                }
                
                # 💾 CACHE SAVE - Lưu direct response vào cache
                try:
                    query_cache.put(request.query, str(request.knowledgeBaseId), config_id, response)
                except Exception as cache_error:
                    logger.warning(f"Failed to cache direct response: {cache_error}")
                
                return response
            
            # Nếu là câu hỏi kiến thức, thực hiện quy trình RAG thông thường
            logger.info(f"[QUERY_SERVICE] Câu hỏi được phân loại là câu hỏi kiến thức, tiếp tục quy trình RAG")
            
            # Sử dụng câu hỏi đã cải thiện (enhanced_query_for_search) cho các bước RAG tiếp theo
            query_for_rag = enhanced_query_for_search
            logger.info(f"[QUERY_SERVICE] Query for RAG (embedding, keyword extraction, reranking): {query_for_rag}")

            logger.info(f"[QUERY_SERVICE] Tải embedding model từ config")
            embedding_model = ModelConfigLoader.load_embedding_model(llm_config)
            
            # Tối ưu 3: Sử dụng RerankerSingleton để tải reranker model trước
            reranker_model_name = llm_config.get("reranker_model", "BAAI/bge-reranker-base")
            logger.info(f"[QUERY_SERVICE] Khởi tạo reranker model singleton cho: {reranker_model_name}")
            reranker_instance = await RerankerSingleton.get_instance(reranker_model_name)
            reranker_tokenizer, reranker_model = None, None
            if reranker_instance:
                reranker_tokenizer, reranker_model = reranker_instance.get_tokenizer_and_model()
                logger.info(f"[QUERY_SERVICE] Đã lấy reranker model từ singleton")
            else:
                logger.warning(f"[QUERY_SERVICE] Không thể khởi tạo reranker model, sẽ bỏ qua quá trình reranking")

            # Chuyển câu hỏi (đã cải thiện) thành embedding với retry mechanism
            logger.info(f"[QUERY_SERVICE] Tạo embedding cho câu hỏi: {query_for_rag}")
            query_embedding = await QueryService._generate_embedding_with_retry(query_for_rag, llm_config)
            # Áp dụng điều chỉnh kích thước cho vector embedding của câu hỏi
            query_embedding = pad_embedding_vector(
                query_embedding, 
                target_dim=embedding_config.target_dimension,
                method=embedding_config.embedding_method
            )
            logger.info(f"[QUERY_SERVICE] Đã tạo embedding có {len(query_embedding)} chiều sau khi điều chỉnh kích thước")

            # Trích xuất từ khóa từ câu hỏi (đã cải thiện)
            # Quyết định phương thức trích xuất từ khóa (ví dụ: từ config hoặc mặc định 'nlp')
            # Giả sử chúng ta có một cấu hình cho việc này, ví dụ: config.get("query_keyword_extraction_method", "nlp")
            # Hiện tại, mặc định dùng NLP. Nếu muốn LLM, cần truyền llm instance (có thể là llm chính)
            query_keywords = await extract_keywords(
                query_for_rag, 
                method="nlp", # Hoặc lấy từ config, ví dụ: config.get("prompt_builder", {}).get("parameters", {}).get("query_keyword_method", "nlp")
                llm_instance=llm # Có thể dùng llm chính để trích xuất keywords nếu method="llm"
            )
            logger.info(f"[QUERY_SERVICE] Extracted keywords from query '{query_for_rag}': {query_keywords}")

            # Tham số tìm kiếm
            params = config["prompt_builder"].get("parameters", {})
            max_chunks = params.get("max_chunks", 10)
            use_reranker = params.get("use_reranker", True)

            # Ghi đè tham số nếu có
            if request.parameters:
                logger.info(f"[QUERY_SERVICE] Override parameters từ request: {json.dumps(request.parameters, indent=2, ensure_ascii=False)}")
                if "max_chunks" in request.parameters:
                    max_chunks = request.parameters["max_chunks"]
                    logger.info(f"[QUERY_SERVICE] Override max_chunks: {max_chunks}")
                if "use_reranker" in request.parameters:
                    use_reranker = request.parameters["use_reranker"]
                    logger.info(f"[QUERY_SERVICE] Override use_reranker: {use_reranker}")

            # Tìm các chunk liên quan trong CSDL - Sử dụng phiên bản đã tối ưu
            logger.info(f"[QUERY_SERVICE] Tìm chunks liên quan, max_chunks={max_chunks}, use_reranker={use_reranker}")
            
            # Tối ưu 4: Truyền tên model reranker và sử dụng singleton
            related_chunks = await QueryService.search_related_chunks(
                query_embedding=query_embedding, # Embedding của câu hỏi đã cải thiện
                query_text=query_for_rag,        # Câu hỏi đã cải thiện (dùng cho reranker)
                query_keywords=query_keywords,   # Từ khóa từ câu hỏi đã cải thiện
                knowledgeBaseId=request.knowledgeBaseId,
                max_chunks=max_chunks,
                use_reranker=use_reranker,
                reranker_tokenizer=reranker_tokenizer,
                reranker_model=reranker_model,
                reranker_model_name=reranker_model_name  # Tối ưu: truyền tên để sử dụng singleton
            )

            if not related_chunks:
                end_time = time.time()
                logger.warning(f"Request {request_id}: No relevant chunks found")
                return {
                    "query": request.query,
                    "llm_answer": "Không tìm thấy thông tin phù hợp để trả lời câu hỏi này.",
                    "model": llm_config.get("model_name"),
                    "config_name": config["name_config"],
                    "config_id": config["id"],
                    "request_id": request_id,
                    "processing_time": end_time - start_time,
                    "retrieved_chunks": []
                }

            logger.info(f"[QUERY_SERVICE] Tìm thấy {len(related_chunks)} chunks liên quan")

            # Lấy thông tin knowledge base
            kb_info = None
            if config["prompt_builder"].get("parameters", {}).get("include_kb_description", True):
                kb_info = await QueryService.get_knowledge_base_info(request.knowledgeBaseId)
                if kb_info:
                    logger.info(f"[QUERY_SERVICE] Đã lấy thông tin knowledge base: {kb_info.get('title', '')}")
                else:
                    logger.warning(f"[QUERY_SERVICE] Không thể lấy thông tin knowledge base")
            
            # Xây dựng prompt và tạo câu trả lời
            context_parts = []
            for chunk in related_chunks:
                # Đảm bảo document_name không rỗng trước khi thêm vào
                doc_name_prefix = f"[Nguồn: {chunk.document_name}] " if chunk.document_name else ""
                context_parts.append(f"{doc_name_prefix}{chunk.chunk_text}")
            context = "\n\n".join(context_parts)
            logger.info(f"[QUERY_SERVICE] Xây dựng prompt với {len(context)} ký tự context, bao gồm tên tài liệu")
            
            final_query_for_llm_prompt = request.query 
            logger.info(f"[QUERY_SERVICE] Query for final LLM prompt: {final_query_for_llm_prompt}")

            if memory:
                custom_params = request.parameters.copy() if request.parameters else {}
                custom_params["conversation_memory"] = memory
                logger.info(f"[QUERY_SERVICE] Đã bổ sung memory vào prompt")
            else:
                custom_params = request.parameters
            
            # Sử dụng prompt builder mới với kb_info nhưng không dùng document_descriptions
            prompt_for_answer = PromptBuilderLoader.build_prompt(
                config["prompt_builder"], 
                final_query_for_llm_prompt, 
                context, 
                custom_params,
                kb_info=kb_info
            )

            # Lấy tham số cho LLM
            llm_params = PromptBuilderLoader.get_parameters(config["prompt_builder"], request.parameters)
            temperature = llm_params.get("temperature", 0.7)
            max_tokens = llm_params.get("max_tokens", 1024)

            # Gọi LLM để tạo câu trả lời
            max_tokens = int(max_tokens) if max_tokens is not None else 1024
            logger.info(f"[QUERY_SERVICE] Gọi LLM với temperature={temperature}, max_tokens={max_tokens}")
            llm_response = llm.generate_content_with_timing(prompt_for_answer, temperature=temperature, max_tokens=max_tokens)
            llm_answer = llm_response["content"]
            llm_time = llm_response.get("processing_time", 0)
            logger.info(f"[QUERY_SERVICE] LLM trả về câu trả lời với {len(llm_answer)} ký tự trong {llm_time:.2f}s")
            
            # Cập nhật memory nếu có chat_section_id - BACKGROUND TASK
            if chat_section_id:
                logger.info(f"[QUERY_SERVICE] Khởi tạo background memory update cho chat_section_id: {chat_section_id}")
                # Background task - không chờ
                asyncio.create_task(QueryService.update_memory_background(
                    chat_section_id, 
                    request.query, 
                    llm_answer, 
                    memory,
                    llm=llm,
                    llm_config=llm_config
                ))

            end_time = time.time()
            processing_time = end_time - start_time

            logger.info(f"Request {request_id}: Query processed in {processing_time:.2f}s")

            # Tạo response với UUID serialization
            # Convert chunks để có thể serialize
            serializable_chunks = []
            for chunk in related_chunks:
                serializable_chunks.append({
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id), 
                    "document_link": chunk.document_link,
                    "chunk_text": chunk.chunk_text,
                    "document_name": chunk.document_name,
                    "similarity_score": chunk.similarity_score,
                    "rerank_score": chunk.rerank_score
                })
            
            response = {
                "query": request.query,
                "retrieved_chunks": serializable_chunks,  # Serializable chunks
                "llm_answer": llm_answer,
                "model": llm_config.get("model_name"),
                "config_name": config["name_config"],
                "config_id": str(config["id"]),    # Convert UUID to string
                "request_id": str(request_id),     # Convert UUID to string
                "processing_time": processing_time,
                "from_cache": False
            }
            
            # 💾 CACHE SAVE - Lưu kết quả vào cache
            try:
                query_cache.put(request.query, str(request.knowledgeBaseId), config_id, response)
            except Exception as cache_error:
                logger.warning(f"Failed to cache result: {cache_error}")
            
            return response

        except Exception as e:
            end_time = time.time()
            logger.error(f"Request {request_id}: Error processing query: {str(e)}", exc_info=True)
            raise

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def search_related_chunks(
        query_embedding: List[float], 
        query_text: str, 
        query_keywords: List[str], 
        knowledgeBaseId: str, 
        max_chunks: int = 10,
        use_reranker: bool = True, 
        reranker_tokenizer: Optional[Any] = None, 
        reranker_model: Optional[Any] = None,
        keyword_search_limit: int = 5, 
        reranker_model_name: str = "BAAI/bge-reranker-base"
    ) -> List[ChunkResponse]:
        """
        Tìm kiếm chunks liên quan sử dụng hybrid search (semantic + keyword) và reranking.
        
        Args:
            query_embedding: Vector embedding của câu truy vấn
            query_text: Văn bản câu truy vấn gốc
            query_keywords: Danh sách keywords được trích xuất
            knowledgeBaseId: ID của knowledge base
            max_chunks: Số lượng chunks tối đa trả về
            use_reranker: Có sử dụng reranker hay không
            reranker_tokenizer: Tokenizer cho reranker
            reranker_model: Model reranker
            keyword_search_limit: Giới hạn cho keyword search
            reranker_model_name: Tên model reranker
            
        Returns:
            List[ChunkResponse]: Danh sách chunks đã được rerank
            
        Raises:
            Exception: Khi có lỗi trong quá trình tìm kiếm
        """
        loop = asyncio.get_event_loop()
        query_vector_list = np.array(query_embedding, dtype=np.float32).tolist() # Chuyển sang list để truyền qua executor

        # Đảm bảo max_chunks là số nguyên
        try:
            max_chunks_int = int(max_chunks)
            logger.info(f"[DB_SEARCH] Converted max_chunks from {type(max_chunks).__name__}: {max_chunks} to int: {max_chunks_int}")
        except (ValueError, TypeError):
            logger.warning(f"[DB_SEARCH] Could not convert max_chunks value: {max_chunks} to integer. Using default value 10.")
            max_chunks_int = 10

        retrieved_chunk_ids = set()
        combined_chunks_for_reranking = []

        try:
            # --- 1. Semantic Search (Vector Search) ---
            # Tối ưu: Giảm limit để ít chunks hơn cần rerank
            semantic_limit = 20  # Giảm từ 15 xuống 10 để rerank nhanh hơn
            logger.info(f"[DB_SEARCH] Performing semantic search with optimized limit: {semantic_limit} (in executor)")
            semantic_rows = await loop.run_in_executor(
                thread_pool_executor,
                QueryService._perform_semantic_search_sync,
                query_vector_list, # Truyền dạng list
                knowledgeBaseId,
                semantic_limit # Sử dụng limit động
            )
            logger.info(f"[DB_SEARCH] Semantic search (executor) found {len(semantic_rows)} chunks.")

            for row in semantic_rows:
                if row[0] not in retrieved_chunk_ids:
                    chunk_id_val = row[0]
                    raw_distance = row[5]
                    
                    distance = float(raw_distance if raw_distance is not None else 1.0)
                    similarity = 1.0 - distance
                    
                    chunk = ChunkResponse(
                        chunk_id=chunk_id_val,
                        document_id=row[1],
                        document_link=row[2],
                        chunk_text=row[3],
                        document_name=row[4],
                        similarity_score=similarity
                    )
                    combined_chunks_for_reranking.append(chunk)
                    retrieved_chunk_ids.add(row[0])
            
            # --- 2. Keyword Search ---
            if query_keywords:
                # Tối ưu: Giảm keyword limit
                keyword_search_limit = min(keyword_search_limit, 3)  # Tối đa 2 chunks từ keyword (total ~12 chunks)
                logger.info(f"[DB_SEARCH] Performing keyword search with keywords: {query_keywords} and optimized limit: {keyword_search_limit} (in executor)")
                keyword_rows = await loop.run_in_executor(
                    thread_pool_executor,
                    QueryService._perform_keyword_search_sync,
                    knowledgeBaseId,
                    query_keywords,
                    keyword_search_limit
                )
                logger.info(f"[DB_SEARCH] Keyword search (executor) found {len(keyword_rows)} chunks.")

                for row in keyword_rows:
                    if row[0] not in retrieved_chunk_ids:
                        chunk = ChunkResponse(
                            chunk_id=row[0],
                            document_id=row[1],
                            document_link=row[2],
                            chunk_text=row[3],
                            document_name=row[4],
                            similarity_score=float(row[5] if row[5] is not None else 0.0)
                        )
                        combined_chunks_for_reranking.append(chunk)
                        retrieved_chunk_ids.add(row[0])
            else:
                logger.info("[DB_SEARCH] No keywords provided for query, skipping keyword search.")

            logger.info(f"[DB_SEARCH] Total unique chunks before reranking: {len(combined_chunks_for_reranking)}")

            if not combined_chunks_for_reranking:
                logger.warning("[DB_SEARCH] No chunks found from either semantic or keyword search.")
                return []

            # --- 3. Rerank combined results ---
            final_reranked_chunks = []
            if use_reranker and combined_chunks_for_reranking:
                logger.info(f"[DB_SEARCH] Reranking {len(combined_chunks_for_reranking)} combined chunks using query: '{query_text}'")
                
                # Tối ưu hóa 1: Sử dụng RerankerSingleton để tránh tải lại model
                reranker_instance = await RerankerSingleton.get_instance(reranker_model_name)
                if reranker_instance:
                    reranker_tokenizer, reranker_model = reranker_instance.get_tokenizer_and_model()
                    logger.info(f"[DB_SEARCH] Sử dụng reranker model đã tải: {reranker_model_name}")
                
                if reranker_tokenizer and reranker_model:
                    # Tối ưu hóa 2: Phân chia chunks thành các batch để xử lý hiệu quả
                    final_reranked_chunks = await QueryService.rerank_chunks_batched(
                        query_text, 
                        combined_chunks_for_reranking, 
                        reranker_tokenizer, 
                        reranker_model
                    )
                    
                    if isinstance(final_reranked_chunks, list):
                        # Sử dụng max_chunks_int đã chuyển đổi ở trên
                        final_reranked_chunks = final_reranked_chunks[:max_chunks_int]
                        logger.info(f"[DB_SEARCH] Reranked to {len(final_reranked_chunks)} chunks.")
                    else:
                        logger.error(f"[DB_SEARCH] Reranking did not return a list as expected. Type: {type(final_reranked_chunks)}. Skipping slicing.")
                else:
                    logger.warning("[DB_SEARCH] Reranker model or tokenizer not available, using combined chunks directly.")
                    # Sử dụng max_chunks_int đã chuyển đổi ở trên
                    final_reranked_chunks = sorted(combined_chunks_for_reranking, 
                                                key=lambda x: x.similarity_score, 
                                                reverse=True)[:max_chunks_int]
            else:
                logger.info("[DB_SEARCH] No reranking performed or reranker not available. Using combined chunks directly.")
                # Sử dụng max_chunks_int đã chuyển đổi ở trên
                final_reranked_chunks = sorted(combined_chunks_for_reranking, 
                                            key=lambda x: x.similarity_score, 
                                            reverse=True)[:max_chunks_int]

            return final_reranked_chunks
        except Exception as e:
            logger.error(f"Error in search_related_chunks: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def _perform_semantic_search_sync(query_vector_list: List[float], knowledgeBaseId: str, limit: int) -> List[Any]:
        conn, pool = None, None
        try:
            conn, pool = get_pg_connection()
            register_vector(conn) # Đăng ký ở đây vì conn là mới cho mỗi lần gọi
            
            cursor = conn.cursor()
            cursor.execute('''
                SELECT chunks.id, chunks.document_id, documents.document_link, chunks.chunk_text, documents.name AS document_name, 
                       (chunks.embedding <=> %s::vector) AS cosine_distance,
                       (chunks.embedding <-> %s::vector) AS l2_distance
                FROM chunks
                JOIN documents ON chunks.document_id = documents.id
                WHERE documents."status" = TRUE
                AND chunks."isDelete" = FALSE
                AND documents."knowledgeBaseId" = %s
                ORDER BY cosine_distance ASC, l2_distance ASC
                LIMIT %s;
            ''', (query_vector_list, query_vector_list, str(knowledgeBaseId), limit))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Sync DB error in _perform_semantic_search_sync: {str(e)}", exc_info=True)
            raise # Re-raise để backoff có thể bắt được nếu cần
        finally:
            if conn:
                return_pg_connection(conn, pool)

    @staticmethod
    def _perform_keyword_search_sync(knowledgeBaseId: str, query_keywords: List[str], limit: int) -> List[Any]:
        conn, pool = None, None
        try:
            conn, pool = get_pg_connection()
            # register_vector không cần thiết cho keyword search thuần túy

            cursor = conn.cursor()
            cursor.execute('''
                SELECT chunks.id, chunks.document_id, documents.document_link, chunks.chunk_text, documents.name AS document_name,
                       0.0 AS keyword_match_score 
                FROM chunks
                JOIN documents ON chunks.document_id = documents.id
                WHERE documents."status" = TRUE
                AND chunks."isDelete" = FALSE
                AND documents."knowledgeBaseId" = %s
                AND chunks.keywords && %s::TEXT[] 
                LIMIT %s; 
            ''', (str(knowledgeBaseId), query_keywords, limit))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Sync DB error in _perform_keyword_search_sync: {str(e)}", exc_info=True)
            raise # Re-raise
        finally:
            if conn:
                return_pg_connection(conn, pool)

    @staticmethod
    def should_skip_reranking(query: str, num_chunks: int) -> bool:
        """Quyết định có nên skip reranking hay không"""
        
        # Skip cho very few chunks
        if num_chunks <= 3:
            logger.info(f"⚡ Skipping reranking for {num_chunks} chunks (≤3, too few)")
            return True
        
        # Skip cho simple greeting queries ONLY (không phải knowledge questions)
        query_lower = query.lower().strip()
        simple_greetings = [
            'xin chào', 'hello', 'hi', 'chào bạn', 'chào em', 'chào anh', 'chào chị',
            'cảm ơn', 'thank you', 'thanks', 'bye', 'tạm biệt',
            'ok', 'được rồi', 'uhm', 'ừm', 'good', 'tốt'
        ]
        
        # Chỉ skip nếu query CHÍNH XÁC match hoặc rất ngắn và đơn giản
        is_simple_greeting = (
            query_lower in simple_greetings or
            (len(query_lower) <= 10 and any(greeting in query_lower for greeting in ['hi', 'ok', 'bye', 'chào', 'cảm ơn']))
        )
        
        if is_simple_greeting:
            logger.info(f"⚡ Skipping reranking for simple greeting: '{query[:20]}...'")
            return True
        
        # Skip cho very short queries (< 5 chars)
        if len(query.strip()) < 5:
            logger.info(f"⚡ Skipping reranking for very short query: '{query}'")
            return True
        
        return False

    @staticmethod
    def get_optimal_thread_count_with_chunk_adjustment(num_chunks: int) -> Tuple[int, int]:
        """
        Tính toán số threads tối ưu và số chunks cần loại bỏ để đảm bảo balanced distribution
        
        Args:
            num_chunks: Số lượng chunks cần rerank
            
        Returns:
            Tuple (optimal_threads, chunks_to_remove)
        """
        if num_chunks <= 3:
            return 1, 0  # Quá ít chunks, dùng 1 thread
        
        if num_chunks % 3 == 0:
            return 3, 0  # Chia hết cho 3 → 3 threads (perfect balance)
        elif num_chunks % 2 == 0:
            return 2, 0  # Chia hết cho 2 → 2 threads (perfect balance)
        else:
            # Không chia hết cho cả 2 và 3 → loại bỏ chunks để chia hết
            # Ưu tiên chia hết cho 3 (hiệu quả hơn) nếu chỉ cần loại bỏ ít chunks
            if (num_chunks - 1) % 3 == 0:
                return 3, 1  # Loại bỏ 1 chunk → chia hết cho 3 → 3 threads
            elif (num_chunks - 1) % 2 == 0:
                return 2, 1  # Loại bỏ 1 chunk → chia hết cho 2 → 2 threads
            elif (num_chunks - 2) % 3 == 0:
                return 3, 2  # Loại bỏ 2 chunks → chia hết cho 3 → 3 threads
            else:
                # Fallback: loại bỏ 1 chunk và dùng 2 threads
                return 2, 1

    @staticmethod
    async def rerank_chunks_batched(
        query: str, 
        chunks: List[ChunkResponse], 
        reranker_tokenizer: Any, 
        reranker_model: Any, 
        batch_size: int = 32
    ) -> List[ChunkResponse]:
        """
        Sử dụng BAAI/bge-reranker-base để rerank với cache và smart optimization
        """
        if not chunks:
            return []
        
        try:
            start_time = time.time()
            num_chunks = len(chunks)
            
            # 🚀 OPTIMIZATION 1: Smart skip logic
            if QueryService.should_skip_reranking(query, num_chunks):
                return chunks
            
            # 🚀 OPTIMIZATION 2: Cache check
            chunk_ids = [str(chunk.chunk_id) for chunk in chunks]
            cached_scores = rerank_cache.get(query, chunk_ids)
            
            if cached_scores:
                # Apply cached scores
                for chunk in chunks:
                    chunk.rerank_score = cached_scores.get(str(chunk.chunk_id), 0.0)
                
                # Sort by cached scores
                sorted_chunks = sorted(chunks, key=lambda x: getattr(x, 'rerank_score', 0.0), reverse=True)
                
                end_time = time.time()
                logger.info(f"🎯 Reranking from cache in {end_time - start_time:.3f}s")
                return sorted_chunks
            
            # 🚀 OPTIMIZATION 3: Calculate optimal threads and chunk removal trước
            optimal_threads, chunks_to_remove = QueryService.get_optimal_thread_count_with_chunk_adjustment(num_chunks)
            
            # Log optimal thread count decision
            logger.info(f"🎯 Optimal thread count for {num_chunks} chunks: {optimal_threads} threads, removing {chunks_to_remove} chunks")
            
            # 🚀 OPTIMIZATION 4: Remove chunks to ensure balanced distribution
            if chunks_to_remove > 0:
                logger.info(f"🎯 Removing {chunks_to_remove} chunks to ensure balanced distribution")
                
                # Sử dụng semantic search scores để loại bỏ chunks kém nhất
                if hasattr(chunks[0], 'similarity_score'):
                    # Sort by semantic score và loại bỏ chunks cuối
                    chunks = sorted(chunks, key=lambda x: getattr(x, 'similarity_score', 0.0), reverse=True)
                    chunks = chunks[:-chunks_to_remove]  # Loại bỏ chunks cuối
                    logger.info(f"🎯 Removed {chunks_to_remove} chunks with lowest semantic scores")
                elif hasattr(chunks[0], 'score'):
                    # Fallback: sử dụng score field
                    chunks = sorted(chunks, key=lambda x: getattr(x, 'score', 0.0), reverse=True)
                    chunks = chunks[:-chunks_to_remove]
                    logger.info(f"🎯 Removed {chunks_to_remove} chunks with lowest scores")
                else:
                    # Nếu không có semantic score, loại bỏ chunks cuối cùng
                    chunks = chunks[:-chunks_to_remove]
                    logger.info(f"🎯 Removed {chunks_to_remove} chunks from end of list")
                
                # Cập nhật num_chunks
                num_chunks = len(chunks)
                logger.info(f"🎯 Updated chunk count: {num_chunks} chunks after removal")
            
            # 🚀 OPTIMIZATION 5: Adaptive batch processing cho BAAI model (sau khi loại bỏ chunks)
            chunk_pairs = [[query, chunk.chunk_text] for chunk in chunks]
            
            # Adaptive batch size dựa trên số chunks - Tối ưu cho BAAI
            if num_chunks <= 8:
                effective_batch_size = num_chunks  # Process all at once
            elif num_chunks <= 16:
                effective_batch_size = 8   # Smaller batch cho BAAI
            else:
                effective_batch_size = 16  # Max batch cho BAAI (thay vì 32)
            
            logger.info(f"🔥 BAAI reranking {num_chunks} chunks with batch_size={effective_batch_size}")
            
            all_scores = []
            
            if optimal_threads == 3:
                # 3 threads processing - perfect balance
                chunk_size = num_chunks // 3
                batch1 = chunk_pairs[:chunk_size]
                batch2 = chunk_pairs[chunk_size:chunk_size*2]
                batch3 = chunk_pairs[chunk_size*2:]
                
                async def process_batch(batch_data):
                    def _predict_sync(pairs):
                        if hasattr(reranker_model, 'predict'):  # SentenceTransformer CrossEncoder
                            return reranker_model.predict(pairs, show_progress_bar=False)
                        else:  # HuggingFace model
                            inputs = reranker_tokenizer(
                                [f"{p[0]} [SEP] {p[1]}" for p in pairs],
                                padding=True, 
                                truncation=True, 
                                return_tensors="pt",
                                max_length=reranker_tokenizer.model_max_length if hasattr(reranker_tokenizer, 'model_max_length') else 512
                            )
                            
                            import torch
                            with torch.no_grad():
                                raw_scores = reranker_model(**inputs).logits
                            
                            return raw_scores.cpu().squeeze().tolist()
                    
                    return await asyncio.get_event_loop().run_in_executor(
                        thread_pool_executor, 
                        lambda: _predict_sync(batch_data)
                    )
                
                # Run all 3 batches in parallel
                logger.info(f"🔀 3x Parallel processing: {len(batch1)} + {len(batch2)} + {len(batch3)} chunks")
                batch1_scores, batch2_scores, batch3_scores = await asyncio.gather(
                    process_batch(batch1),
                    process_batch(batch2),
                    process_batch(batch3)
                )
                
                # Combine scores from all 3 batches
                for batch_scores in [batch1_scores, batch2_scores, batch3_scores]:
                    if isinstance(batch_scores, (list, tuple)):
                        all_scores.extend(batch_scores)
                    else:
                        all_scores.append(batch_scores)
                        
            elif optimal_threads == 2:
                # 2 threads processing - perfect balance
                chunk_size = num_chunks // 2
                batch1 = chunk_pairs[:chunk_size]
                batch2 = chunk_pairs[chunk_size:]
                
                async def process_batch(batch_data):
                    def _predict_sync(pairs):
                        if hasattr(reranker_model, 'predict'):  # SentenceTransformer CrossEncoder
                            return reranker_model.predict(pairs, show_progress_bar=False)
                        else:  # HuggingFace model
                            inputs = reranker_tokenizer(
                                [f"{p[0]} [SEP] {p[1]}" for p in pairs],
                                padding=True, 
                                truncation=True, 
                                return_tensors="pt",
                                max_length=reranker_tokenizer.model_max_length if hasattr(reranker_tokenizer, 'model_max_length') else 512
                            )
                            
                            import torch
                            with torch.no_grad():
                                raw_scores = reranker_model(**inputs).logits
                            
                            return raw_scores.cpu().squeeze().tolist()
                    
                    return await asyncio.get_event_loop().run_in_executor(
                        thread_pool_executor, 
                        lambda: _predict_sync(batch_data)
                    )
                
                # Run 2 batches in parallel
                logger.info(f"🔀 2x Parallel processing: {len(batch1)} + {len(batch2)} chunks")
                batch1_scores, batch2_scores = await asyncio.gather(
                    process_batch(batch1),
                    process_batch(batch2)
                )
                
                # Combine scores from both batches
                for batch_scores in [batch1_scores, batch2_scores]:
                    if isinstance(batch_scores, (list, tuple)):
                        all_scores.extend(batch_scores)
                    else:
                        all_scores.append(batch_scores)
                        
            else:  # optimal_threads == 1
                # Single thread processing - stable but slower
                logger.info(f"🔥 Single thread processing: {num_chunks} chunks")
                # Sequential processing cho ít chunks
                for i in range(0, num_chunks, effective_batch_size):
                    batch = chunk_pairs[i:i+effective_batch_size]
                    logger.info(f"📦 Processing batch {i//effective_batch_size + 1} with {len(batch)} pairs")
                    
                    def _predict_batch_scores_sync(batch_pairs):
                        if hasattr(reranker_model, 'predict'):  # SentenceTransformer CrossEncoder
                            return reranker_model.predict(batch_pairs, show_progress_bar=False)
                        else:  # HuggingFace model
                            inputs = reranker_tokenizer(
                                [f"{p[0]} [SEP] {p[1]}" for p in batch_pairs],
                                padding=True, 
                                truncation=True, 
                                return_tensors="pt",
                                max_length=reranker_tokenizer.model_max_length if hasattr(reranker_tokenizer, 'model_max_length') else 512
                            )
                            
                            import torch
                            with torch.no_grad():
                                raw_scores = reranker_model(**inputs).logits
                            
                            return raw_scores.cpu().squeeze().tolist()
                    
                    batch_scores = await asyncio.get_event_loop().run_in_executor(
                        thread_pool_executor, 
                        lambda: _predict_batch_scores_sync(batch)
                    )
                    
                    # Handle single score vs list of scores
                    if isinstance(batch_scores, (list, tuple)):
                        all_scores.extend(batch_scores)
                    else:
                        all_scores.append(batch_scores)
            
            # Validate scores
            if len(all_scores) != len(chunks):
                logger.error(f"❌ Score mismatch: {len(all_scores)} scores vs {len(chunks)} chunks")
                return chunks
            
            # 🚀 OPTIMIZATION 5: Apply scores and cache results
            scores_dict = {}
            for i, chunk in enumerate(chunks):
                try:
                    score = float(all_scores[i])
                    chunk.rerank_score = score
                    scores_dict[str(chunk.chunk_id)] = score
                except (TypeError, ValueError) as e:
                    logger.error(f"❌ Score conversion error for chunk {chunk.chunk_id}: {e}")
                    chunk.rerank_score = -float('inf')
                    scores_dict[str(chunk.chunk_id)] = -float('inf')
            
            # Cache the results
            rerank_cache.put(query, chunk_ids, scores_dict)
            
            # Sort by scores
            sorted_chunks = sorted(chunks, key=lambda x: getattr(x, 'rerank_score', -float('inf')), reverse=True)
            
            end_time = time.time()
            rerank_time = end_time - start_time
            logger.info(f"✅ BAAI reranking completed in {rerank_time:.2f}s (avg: {rerank_time/num_chunks:.3f}s/chunk)")
            
            return sorted_chunks
            
        except Exception as e:
            logger.error(f"❌ Reranking error: {e}", exc_info=True)
            return chunks

    @staticmethod
    async def rerank_chunks(query: str, chunks: List[ChunkResponse], reranker_tokenizer: Any, reranker_model: Any) -> List[ChunkResponse]:
        """
        Legacy rerank_chunks function - now redirects to the batched version for backward compatibility
        """
        return await QueryService.rerank_chunks_batched(query, chunks, reranker_tokenizer, reranker_model)

    @staticmethod
    async def get_memory(chat_section_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy memory của một chat section từ database.
        
        Args:
            chat_section_id: ID của chat section
            
        Returns:
            Đối tượng memory dạng dict hoặc None nếu không tìm thấy
        """
        try:
            conn, pool = get_pg_connection()
            
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT memory FROM chat_sections 
                    WHERE id = %s AND "isDeleted" = FALSE;
                """, (str(chat_section_id),))
                
                result = cursor.fetchone()
                
                if result and result[0]:
                    logger.info(f"[QUERY_SERVICE] Đã lấy memory cho chat_section_id: {chat_section_id}")
                    return result[0]  # PostgreSQL tự động chuyển đổi JSONB thành dict
                else:
                    logger.info(f"[QUERY_SERVICE] Không tìm thấy memory cho chat_section_id: {chat_section_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error getting memory for chat_section_id {chat_section_id}: {str(e)}", exc_info=True)
                return None
            finally:
                return_pg_connection(conn, pool)
        except Exception as e:
            logger.error(f"Connection error in get_memory: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    async def get_knowledge_base_info(knowledge_base_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chi tiết về knowledge base từ database.
        
        Args:
            knowledge_base_id: ID của knowledge base
            
        Returns:
            Thông tin knowledge base dạng dict hoặc None nếu không tìm thấy
        """
        try:
            conn, pool = get_pg_connection()
            
            try:
                # Truy vấn lấy thông tin từ knowledge_base
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, description, created_by, updated_at
                    FROM knowledge_base
                    WHERE id = %s AND "isDeleted" = FALSE;
                """, (str(knowledge_base_id),))
                
                result = cursor.fetchone()
                
                if result:
                    logger.info(f"[QUERY_SERVICE] Đã lấy thông tin knowledge base ID: {knowledge_base_id}")
                    
                    kb_info = {
                        "id": str(result[0]),
                        "title": result[1],
                        "description": result[2],
                        "created_by": result[3],
                        "updated_at": result[4].isoformat() if result[4] else None
                    }
                    
                    return kb_info
                else:
                    logger.info(f"[QUERY_SERVICE] Không tìm thấy knowledge base ID: {knowledge_base_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error getting knowledge base info for ID {knowledge_base_id}: {str(e)}", exc_info=True)
                return None
            finally:
                return_pg_connection(conn, pool)
        except Exception as e:
            logger.error(f"Connection error in get_knowledge_base_info: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    async def update_memory_background(
        chat_section_id: str, 
        current_query: str, 
        llm_answer: str, 
        existing_memory: Optional[Dict[str, Any]] = None,
        llm=None,
        llm_config=None
    ) -> None:
        """Background memory update - không block main thread"""
        try:
            await QueryService.update_memory(
                chat_section_id, current_query, llm_answer, 
                existing_memory, llm, llm_config
            )
            logger.info(f"[QUERY_SERVICE] Background memory update completed for: {chat_section_id}")
        except Exception as e:
            logger.error(f"[QUERY_SERVICE] Background memory update failed for {chat_section_id}: {e}")
    
    @staticmethod
    async def _save_conversation_history(chat_section_id: str, user_query: str, assistant_response: str):
        """Lưu conversation history để tối ưu token"""
        try:
            from services.conversation_manager import conversation_manager
            
            # Lưu user message
            conversation_manager.add_message(chat_section_id, "user", user_query)
            
            # Lưu assistant response
            conversation_manager.add_message(chat_section_id, "assistant", assistant_response)
            
            logger.info(f"[QUERY_SERVICE] Đã lưu conversation history cho {chat_section_id}")
        except Exception as e:
            logger.error(f"[QUERY_SERVICE] Lỗi khi lưu conversation history cho {chat_section_id}: {e}")

    @staticmethod
    async def update_memory(
        chat_section_id: str, 
        current_query: str, 
        llm_answer: str, 
        existing_memory: Optional[Dict[str, Any]] = None,
        llm=None,
        llm_config=None
    ) -> Optional[Dict[str, Any]]:
        """
        Cập nhật memory của một chat section.
        
        Args:
            chat_section_id: ID của chat section
            current_query: Câu hỏi hiện tại của người dùng
            llm_answer: Câu trả lời của LLM
            existing_memory: Memory hiện tại (nếu đã tải trước đó)
            llm: Model LLM đã được tải (nếu có)
            llm_config: Cấu hình LLM (nếu có)
            
        Returns:
            Memory đã cập nhật dạng dict hoặc None nếu có lỗi
        """
        try:
            # Nếu chưa có existing_memory, lấy từ database
            if existing_memory is None:
                existing_memory = await QueryService.get_memory(chat_section_id)
            
            # Nếu vẫn không có memory, tạo mới
            if existing_memory is None:
                existing_memory = QueryService.create_initial_memory()
            
            # Sử dụng model đã được truyền vào hoặc tải model mới nếu không có
            if llm is None:
                # Tải LLM phù hợp
                from llms.config_loader import ModelConfigLoader
                
                # Nếu có llm_config, sử dụng cấu hình đó, nếu không sử dụng cấu hình mặc định
                if llm_config is None:
                    logger.info(f"[QUERY_SERVICE] Không có LLM config, sử dụng model mặc định")
                    llm_config = {
                        "model_type": "online",
                        "model_name": "gemini-1.5-flash-latest",
                        "api_key": None  # Sẽ lấy từ biến môi trường
                    }
                else:
                    logger.info(f"[QUERY_SERVICE] Sử dụng LLM config từ cấu hình người dùng: {llm_config.get('model_name')}")
                
                llm = ModelConfigLoader.load_model(llm_config)
                logger.info(f"[QUERY_SERVICE] Đã tải model {llm_config.get('model_name')} cho cập nhật memory")
            else:
                logger.info(f"[QUERY_SERVICE] Sử dụng model hiện có cho cập nhật memory")
            
            # Tạo prompt để cập nhật memory
            prompt = f"""
            Dưới đây là thông tin memory hiện tại về cuộc hội thoại:
            {json.dumps(existing_memory, indent=2, ensure_ascii=False)}
            
            Tin nhắn mới:
            Người dùng: {current_query}
            AI: {llm_answer}
            
            Hãy cập nhật thông tin memory theo cấu trúc JSON sau:
            1. Cập nhật trường "summary" để tóm tắt toàn bộ cuộc hội thoại
            2. Cập nhật "topics" nếu có chủ đề mới
            3. Thêm entities mới được nhắc đến
            4. Cập nhật "key_points" với những điểm quan trọng mới
            5. Thêm câu hỏi hiện tại vào "last_questions"
            6. Cập nhật "context.current_focus" nếu chủ đề thay đổi
            7. Cập nhật "metadata"
            
            Trả về JSON cập nhật đầy đủ, chỉ bao gồm thông tin quan trọng và ngắn gọn.
            Đảm bảo trả về chuỗi JSON hợp lệ, không có dữ liệu khác.
            """
            
            # Gọi LLM để cập nhật memory
            llm_response = llm.generate_content(prompt)
            
            try:
                # Xử lý dữ liệu phản hồi từ LLM, hỗ trợ cả định dạng dict và string
                if isinstance(llm_response, dict) and "content" in llm_response:
                    content = llm_response["content"]
                elif isinstance(llm_response, str):
                    content = llm_response
                else:
                    logger.error(f"Unexpected response format from LLM: {type(llm_response)}")
                    return await QueryService.update_memory_manually(chat_section_id, current_query, llm_answer, existing_memory)
                
                # Tìm và trích xuất phần JSON
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if json_match:
                    content = json_match.group(1)
                
                # Parse JSON từ response của LLM
                new_memory = json.loads(content)
                
                # Validate JSON structure
                QueryService.validate_memory_structure(new_memory)
                
                # Cập nhật memory vào database
                conn, pool = get_pg_connection()
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        UPDATE chat_sections SET memory = %s
                        WHERE id = %s;
                    """, (json.dumps(new_memory), str(chat_section_id)))
                    
                    conn.commit()
                    logger.info(f"[QUERY_SERVICE] Đã cập nhật memory cho chat_section_id: {chat_section_id}")
                    return new_memory
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Database error updating memory for chat_section_id {chat_section_id}: {str(e)}", exc_info=True)
                    return existing_memory
                finally:
                    return_pg_connection(conn, pool)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in LLM response: {str(e)}\nResponse: {content[:200]}...", exc_info=True)
                
                # Fallback: Cập nhật memory thủ công
                return await QueryService.update_memory_manually(chat_section_id, current_query, llm_answer, existing_memory)
                
        except Exception as e:
            logger.error(f"Error updating memory for chat_section_id {chat_section_id}: {str(e)}", exc_info=True)
            return await QueryService.update_memory_manually(chat_section_id, current_query, llm_answer, existing_memory)
    
    @staticmethod
    def create_initial_memory() -> Dict[str, Any]:
        """
        Tạo cấu trúc memory ban đầu
        """
        return {
            "summary": "",
            "topics": [],
            "entities": {
                "people": [],
                "organizations": [],
                "locations": []
            },
            "key_points": [],
            "last_questions": [],
            "context": {
                "current_focus": "",
                "references": []
            },
            "metadata": {
                "message_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def validate_memory_structure(memory: Dict[str, Any]) -> bool:
        """
        Kiểm tra cấu trúc memory có hợp lệ không
        """
        required_keys = ["summary", "topics", "key_points", "last_questions", "metadata"]
        for key in required_keys:
            if key not in memory:
                logger.warning(f"Missing required key in memory: {key}")
                memory[key] = [] if key in ["topics", "key_points", "last_questions"] else {}
                if key == "summary":
                    memory[key] = ""
                    
        # Đảm bảo metadata
        if "metadata" not in memory or not isinstance(memory["metadata"], dict):
            memory["metadata"] = {}
            
        if "message_count" not in memory["metadata"]:
            memory["metadata"]["message_count"] = 0
            
        if "last_updated" not in memory["metadata"]:
            memory["metadata"]["last_updated"] = datetime.now().isoformat()
            
        return True
    
    @staticmethod
    async def update_memory_manually(chat_section_id: str, current_query: str, llm_answer: str, 
                                   existing_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cập nhật memory thủ công trong trường hợp LLM không trả về JSON hợp lệ
        """
        try:
            # Đảm bảo cấu trúc hợp lệ
            QueryService.validate_memory_structure(existing_memory)
            
            # Cập nhật các trường cơ bản
            existing_memory["summary"] += f" User hỏi về: {current_query[:50]}..."
            
            # Thêm câu hỏi vào last_questions
            existing_memory["last_questions"].append({
                "question": current_query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Giới hạn số lượng câu hỏi lưu trữ
            if len(existing_memory["last_questions"]) > 5:
                existing_memory["last_questions"] = existing_memory["last_questions"][-5:]
            
            # Cập nhật metadata
            existing_memory["metadata"]["message_count"] += 2  # +2 cho cả câu hỏi và trả lời
            existing_memory["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Lưu memory vào database
            conn, pool = get_pg_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    UPDATE chat_sections SET memory = %s
                    WHERE id = %s;
                """, (json.dumps(existing_memory), str(chat_section_id)))
                
                conn.commit()
                logger.info(f"[QUERY_SERVICE] Đã cập nhật memory thủ công cho chat_section_id: {chat_section_id}")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error in manual memory update for chat_section_id {chat_section_id}: {str(e)}", exc_info=True)
            finally:
                return_pg_connection(conn, pool)
                
            return existing_memory
            
        except Exception as e:
            logger.error(f"Error in manual memory update for chat_section_id {chat_section_id}: {str(e)}", exc_info=True)
            return existing_memory