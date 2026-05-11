import json
import logging
from fastapi import HTTPException
from llms.onlinellms import OnlineLLMs
from llms.localLllms import LocalLlms
from llms.local_embedding import LocalEmbedding
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# --- simple round-robin for api_keys ---
_RR_IDX = {}
_KEY_COOLDOWN = {}  # api_key -> epoch seconds

import time

def reset_cooldowns():
    """Reset tất cả cooldown khi restart server"""
    global _KEY_COOLDOWN, _RR_IDX
    _KEY_COOLDOWN.clear()
    _RR_IDX.clear()
    logger.info("[CONFIG_LOADER] Đã reset tất cả cooldown và round-robin index")

def mark_rate_limited(key: str, seconds: int = 120):
    """Đánh dấu một API key đang bị rate-limit để bỏ qua tạm thời (mặc định 2 phút)."""
    if key:
        _KEY_COOLDOWN[key] = time.time() + seconds

def _is_cooldown(key: str) -> bool:
    exp = _KEY_COOLDOWN.get(key, 0)
    return bool(exp and exp > time.time())

def _get_soonest_available_key(keys):
    """Chọn key có thời gian cooldown còn lại ngắn nhất để fallback.
    
    Trả về None nếu danh sách keys rỗng.
    """
    if not keys:
        return None
    now = time.time()
    # (remaining_seconds, key)
    candidates = []
    for k in keys:
        exp = _KEY_COOLDOWN.get(k, 0)
        remaining = max(0, exp - now) if exp and exp > now else 0
        candidates.append((remaining, k))
    # Ưu tiên key đã hết cooldown (remaining=0), nếu không có thì chọn key còn ít giây nhất
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def pick_api_key(llm_cfg: dict):
    keys = llm_cfg.get("api_keys") or []
    single = llm_cfg.get("api_key")
    if isinstance(keys, list) and keys:
        model_id = f"{llm_cfg.get('model_type')}::{llm_cfg.get('model_name')}"
        # Vòng quay có skip các key đang cooldown
        for _ in range(len(keys)):
            i = _RR_IDX.get(model_id, 0) % len(keys)
            _RR_IDX[model_id] = i + 1
            k = keys[i]
            if not _is_cooldown(k):
                return k
        # Nếu tất cả key đều cooldown, chọn key gần hết cooldown nhất
        return _get_soonest_available_key(keys)
    return single


class ModelConfigLoader:
    _loaded_models = {}
    _loaded_embedding_models = {}
    _loaded_rerankers = {}

    @staticmethod
    def load_model(llm_config):
        """Tải mô hình từ cấu hình"""
        try:
            # Tạo key cho cache dựa trên các tham số quan trọng của config
            model_type = llm_config.get("model_type")
            model_name = llm_config.get("model_name")
            # pick API key (prefer array api_keys)
            api_key = pick_api_key(llm_config)
            endpoint = llm_config.get("endpoint")
            
            # Cache key không bao gồm API key để cho phép round-robin hoạt động
            cache_key = (model_type, model_name, endpoint)

            # Không sử dụng cache cho LLM models để cho phép round-robin API keys
            # Mỗi lần gọi sẽ pick API key mới và tạo model instance mới
            logger.info(f"[CONFIG_LOADER] Tạo LLM model mới với round-robin API key")

            # Log toàn bộ cấu hình (che API key nếu có)
            config_log = llm_config.copy() if isinstance(llm_config, dict) else {}
            if "api_key" in config_log and config_log["api_key"]:
                # Hiển thị 3 ký tự đầu và 3 ký tự cuối của API key
                api_key_val = config_log["api_key"] # Sử dụng biến mới để tránh nhầm lẫn với api_key ở trên
                masked_key = f"{api_key_val[:3]}...{api_key_val[-3:]}" if len(api_key_val) > 6 else "***"
                config_log["api_key"] = masked_key
            # Ẩn api_keys nếu có
            if "api_keys" in config_log and isinstance(config_log["api_keys"], list):
                config_log["api_keys"] = [f"{k[:3]}...{k[-3:]}" if isinstance(k, str) and len(k) > 6 else "***" for k in config_log["api_keys"]]
            
            logger.info(f"[CONFIG_LOADER] Đang tải cấu hình mô hình từ DB: {json.dumps(config_log, indent=2, ensure_ascii=False)}")
            
            logger.info(f"[CONFIG_LOADER] Tải model: {model_name} (loại: {model_type})")
            logger.info(f"[CONFIG_LOADER] Sử dụng endpoint: {endpoint or 'mặc định'}")
            logger.info(f"[CONFIG_LOADER] API key: {'Đã cung cấp' if api_key else 'Không có'}")

            # Validate API key presence for online models
            if model_type == "online" and not api_key:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Vui lòng kiểm tra lại cấu hình: chưa có API key hợp lệ (api_keys rỗng hoặc api_key trống)."
                    ),
                )

            model_instance = None
            if model_type == "online":
                # Khởi tạo mô hình online với API key từ cấu hình
                logger.info(f"[CONFIG_LOADER] Khởi tạo Online LLM với model: {model_name}")
                model_instance = OnlineLLMs(model_name=model_name, api_key=api_key, endpoint=endpoint)
            elif model_type == "local":
                # Khởi tạo mô hình local với endpoint từ cấu hình nếu có
                logger.info(f"[CONFIG_LOADER] Khởi tạo Local LLM với model: {model_name}")
                model_instance = LocalLlms(model_name=model_name, endpoint=endpoint)
            else:
                logger.error(f"[CONFIG_LOADER] Không hỗ trợ loại mô hình: {model_type}")
                raise ValueError(f"Không hỗ trợ loại mô hình: {model_type}")
            
            # Không cache LLM models để cho phép round-robin
            return model_instance
        except Exception as e:
            logger.error(f"[CONFIG_LOADER] Lỗi khi tải model: {str(e)}")
            raise

    @staticmethod
    def load_embedding_model(llm_config):
        """Tải mô hình embedding từ cấu hình"""
        try:
            # Sử dụng embedding_type để phân biệt, fallback về model_type nếu không có
            embedding_type = llm_config.get("embedding_type", llm_config.get("model_type", "online"))
            embedding_model_name_config = llm_config.get("embedding_model")
            embedding_provider = llm_config.get("embedding_provider")
            # pick API key for embeddings as well
            api_key = pick_api_key(llm_config)
            endpoint = llm_config.get("endpoint")
            
            # Xác định embedding_model_name thực tế sẽ được sử dụng
            actual_embedding_model_name = embedding_model_name_config
            if embedding_type == "online":
                if not embedding_model_name_config:
                    actual_embedding_model_name = "text-embedding-3-small" # Fallback cho online
            else: # local
                # Cho local, sử dụng SentenceTransformer model
                actual_embedding_model_name = embedding_model_name_config or "all-MiniLM-L6-v2"

            # Cache key không bao gồm API key để cho phép round-robin hoạt động
            # Chỉ cache dựa trên model config, không phải API key cụ thể
            cache_key = (embedding_type, actual_embedding_model_name, endpoint, embedding_provider)

            # Kiểm tra cache trước khi tạo model mới
            if cache_key in ModelConfigLoader._loaded_embedding_models:
                logger.info(f"[CONFIG_LOADER] Trả về embedding model đã cache: {actual_embedding_model_name}")
                return ModelConfigLoader._loaded_embedding_models[cache_key]
            
            logger.info(f"[CONFIG_LOADER] Tạo embedding model mới: {actual_embedding_model_name}")

            logger.info(f"[CONFIG_LOADER] Tải embedding model: {embedding_model_name_config or 'không cung cấp (sẽ fallback)'} (type: {embedding_type}, provider: {embedding_provider or 'auto'})")
            # Không dùng endpoint local cho embeddings nếu embedding_type là local
            embedding_endpoint = endpoint if embedding_type == 'online' else None
            logger.info(f"[CONFIG_LOADER] Sử dụng endpoint embedding: {embedding_endpoint or 'mặc định'}")
            logger.info(f"[CONFIG_LOADER] API key cho embedding: {'Đã cung cấp' if api_key else 'Không có'}")   

            if embedding_type == "online" and not api_key:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Vui lòng kiểm tra lại cấu hình embedding: chưa có API key hợp lệ (api_keys rỗng hoặc api_key trống)."
                    ),
                )

            embedding_instance = None
            if embedding_type == "online":
                if embedding_model_name_config:
                    logger.info(f"[CONFIG_LOADER] Khởi tạo embedding model online: {embedding_model_name_config}")
                    embedding_instance = OnlineLLMs(model_name=embedding_model_name_config, api_key=api_key, endpoint=embedding_endpoint, provider=embedding_provider)
                else:
                    logger.warning(f"[CONFIG_LOADER] Không tìm thấy model embedding trong cấu hình, sử dụng mặc định {actual_embedding_model_name}")
                    embedding_instance = OnlineLLMs(model_name=actual_embedding_model_name, api_key=api_key, endpoint=embedding_endpoint, provider=embedding_provider)
            else:  # local
                logger.info(f"[CONFIG_LOADER] Khởi tạo local embedding model: {actual_embedding_model_name}")
                embedding_instance = LocalEmbedding(model_name=actual_embedding_model_name)
            
            # Không cache embedding models để cho phép round-robin
            return embedding_instance
        except Exception as e:
            logger.error(f"[CONFIG_LOADER] Lỗi khi tải embedding model: {str(e)}")
            raise

    @staticmethod
    def load_reranker(llm_config):
        """Tải mô hình reranker từ cấu hình"""
        try:
            reranker_model_name = llm_config.get("reranker_model", "BAAI/bge-reranker-base")
            cache_key = reranker_model_name # Reranker chỉ phụ thuộc vào tên

            if cache_key in ModelConfigLoader._loaded_rerankers:
                logger.info(f"[CONFIG_LOADER] Trả về reranker đã cache: {reranker_model_name}")
                return ModelConfigLoader._loaded_rerankers[cache_key]

            logger.info(f"[CONFIG_LOADER] Tải reranker model: {reranker_model_name}")

            tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            
            ModelConfigLoader._loaded_rerankers[cache_key] = (tokenizer, model)
            return tokenizer, model
        except Exception as e:
            logger.error(f"[CONFIG_LOADER] Lỗi khi tải reranker model: {str(e)}")
            logger.warning("[CONFIG_LOADER] Tiếp tục mà không có reranker")
            return None, None

    @staticmethod
    def load_reranker_sync(model_name="BAAI/bge-reranker-base"):
        """
        Phiên bản đồng bộ của load_reranker được thiết kế để chạy trong ThreadPoolExecutor.
        Tải mô hình reranker theo tên, sử dụng cache nếu đã tải trước đó.
        """
        try:
            cache_key = model_name # Reranker chỉ phụ thuộc vào tên

            if cache_key in ModelConfigLoader._loaded_rerankers:
                logger.info(f"[CONFIG_LOADER] Trả về reranker đã cache: {model_name}")
                return ModelConfigLoader._loaded_rerankers[cache_key]

            logger.info(f"[CONFIG_LOADER] Tải reranker model (đồng bộ): {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Lưu vào cache chung với phiên bản bất đồng bộ
            ModelConfigLoader._loaded_rerankers[cache_key] = (tokenizer, model)
            return tokenizer, model
        except Exception as e:
            logger.error(f"[CONFIG_LOADER] Lỗi khi tải reranker model (đồng bộ): {str(e)}")
            logger.warning("[CONFIG_LOADER] Tiếp tục mà không có reranker")
            return None, None


class PromptBuilderLoader:
    @staticmethod
    def build_prompt(prompt_builder, query, context, parameters=None, kb_info=None):
        """
        Xây dựng prompt từ cấu hình prompt_builder với khả năng ghi đè từ parameters.
        
        Args:
            prompt_builder: Cấu hình prompt
            query: Câu hỏi
            context: Context cho prompt
            parameters: Tham số tùy chỉnh
            kb_info: Thông tin knowledge base
        
        Returns:
            Prompt hoàn chỉnh (legacy format)
        """
        # Sử dụng build_messages và chuyển đổi về format cũ
        messages = PromptBuilderLoader.build_messages(prompt_builder, query, context, parameters, kb_info)
        return PromptBuilderLoader._messages_to_prompt(messages)
    
    @staticmethod
    def build_messages(prompt_builder, query, context, parameters=None, kb_info=None, conversation_history=None, chat_section_id=None):
        """
        Xây dựng messages format tối ưu cho LLM với system/user separation và conversation history.
        
        Args:
            prompt_builder: Cấu hình prompt
            query: Câu hỏi
            context: Context cho prompt
            parameters: Tham số tùy chỉnh
            kb_info: Thông tin knowledge base
            conversation_history: Lịch sử cuộc trò chuyện (legacy)
            chat_section_id: ID của chat section để quản lý conversation
        
        Returns:
            List[Dict]: Messages format cho LLM
        """
        try:
            # Lấy giá trị mặc định từ prompt_builder
            system = prompt_builder.get("system_instruction", "")
            context_template = prompt_builder.get("context_template", "")
            query_template = prompt_builder.get("query_template", "")
            instruction = prompt_builder.get("instruction_template", "")
            
            # Xác định xem có sử dụng thông tin kb không
            use_kb_info = prompt_builder.get("parameters", {}).get("include_kb_description", True)

            # Ghi đè bằng giá trị tùy chỉnh từ parameters nếu có
            if parameters:
                if "custom_system_instruction" in parameters and parameters["custom_system_instruction"].strip():
                    custom_system = parameters["custom_system_instruction"]
                    logger.info(f"[PROMPT_BUILDER] Sử dụng system_instruction tùy chỉnh: {custom_system[:50]}...")
                    system = custom_system
                
                if "custom_instruction_template" in parameters and parameters["custom_instruction_template"].strip():
                    custom_instruction = parameters["custom_instruction_template"]
                    logger.info(f"[PROMPT_BUILDER] Sử dụng instruction_template tùy chỉnh: {custom_instruction[:50]}...")
                    instruction = custom_instruction
                    
                # Ghi đè tham số include_kb_description nếu có
                if "include_kb_description" in parameters:
                    use_kb_info = parameters["include_kb_description"]
                    logger.info(f"[PROMPT_BUILDER] Override include_kb_description: {use_kb_info}")

            # Xử lý và tích hợp dữ liệu memory vào system instruction
            memory_section = ""
            if parameters and "conversation_memory" in parameters:
                memory = parameters["conversation_memory"]
                logger.info(f"[PROMPT_BUILDER] Tích hợp conversation_memory vào system instruction")
                
                # Tạo section tóm tắt cuộc hội thoại từ memory
                memory_section = "### LỊCH SỬ CUỘC HỘI THOẠI:\n"
                
                # Thêm tóm tắt
                if "summary" in memory and memory["summary"]:
                    memory_section += f"Tóm tắt: {memory['summary']}\n\n"
                
                # Thêm chủ đề đã thảo luận
                if "topics" in memory and memory["topics"]:
                    memory_section += f"Chủ đề: {', '.join(memory['topics'])}\n\n"
                
                # Thêm các điểm chính đã đề cập
                if "key_points" in memory and memory["key_points"]:
                    memory_section += "Các điểm chính đã đề cập:\n"
                    for point in memory["key_points"]:
                        memory_section += f"- {point}\n"
                    memory_section += "\n"
                
                # Thêm các câu hỏi trước đó
                if "last_questions" in memory and memory["last_questions"]:
                    memory_section += "Các câu hỏi gần đây:\n"
                    for q_data in memory["last_questions"][-3:]:  # Chỉ lấy tối đa 3 câu hỏi gần nhất
                        if isinstance(q_data, dict) and "question" in q_data:
                            memory_section += f"- {q_data['question']}\n"
                        else:
                            memory_section += f"- {q_data}\n"
                    memory_section += "\n"
                
                # Thêm focus hiện tại
                if "context" in memory and "current_focus" in memory["context"] and memory["context"]["current_focus"]:
                    memory_section += f"Trọng tâm hiện tại: {memory['context']['current_focus']}\n\n"
                    
                logger.info(f"[PROMPT_BUILDER] Đã tạo phần memory_section với {len(memory_section)} ký tự")
            
            # Tạo phần kb_info_section nếu có kb_info và use_kb_info=True
            kb_info_section = ""
            if kb_info and use_kb_info:
                logger.info(f"[PROMPT_BUILDER] Tích hợp thông tin Knowledge Base vào system instruction")
                kb_info_section = "### THÔNG TIN CƠ SỞ KIẾN THỨC:\n"
                
                # Thêm tiêu đề
                if kb_info.get("title"):
                    kb_info_section += f"Tiêu đề: {kb_info['title']}\n\n"
                
                # Thêm mô tả
                if kb_info.get("description"):
                    kb_info_section += f"Mô tả: {kb_info['description']}\n\n"
                
                logger.info(f"[PROMPT_BUILDER] Đã tạo phần kb_info_section với {len(kb_info_section)} ký tự")

            # Xây dựng user message trước
            context_section = context_template.replace("{{context}}", context)
            query_section = query_template.replace("{{query}}", query)
            user_content = f"{context_section}\n\n{query_section}\n\n{instruction}"
            
            # Kiểm tra conversation history optimization TRƯỚC
            if chat_section_id:
                from services.conversation_manager import conversation_manager
                
                # Kiểm tra xem có conversation history hay không
                if conversation_manager.has_conversation_history(chat_section_id):
                    # Có conversation history, sử dụng optimized messages
                    # Chỉ cập nhật system message nếu có memory mới
                    existing_system = conversation_manager.get_system_message(chat_section_id)
                    if memory_section and existing_system:
                        # Cập nhật system message với memory mới
                        updated_system = existing_system['content'].replace("{{conversation_memory}}", memory_section)
                        conversation_manager.set_system_message(chat_section_id, updated_system)
                        logger.info(f"[PROMPT_BUILDER] Cập nhật system message với memory mới cho {chat_section_id}")
                    
                    optimized_messages = conversation_manager.build_optimized_messages(chat_section_id, user_content)
                    logger.info(f"[PROMPT_BUILDER] Sử dụng conversation history tối ưu cho {chat_section_id}")
                    return optimized_messages
                else:
                    # Lần đầu, tạo system message và lưu
                    system_content = system
                    if memory_section:
                        system_content = system.replace("{{conversation_memory}}", memory_section)
                    
                    if kb_info_section:
                        system_content = f"{system_content}\n\n{kb_info_section}"
                    
                    conversation_manager.set_system_message(chat_section_id, system_content)
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                    logger.info(f"[PROMPT_BUILDER] Tạo conversation mới cho {chat_section_id}")
            else:
                # Fallback: tạo messages thông thường
                system_content = system
                if memory_section:
                    system_content = system.replace("{{conversation_memory}}", memory_section)
                
                if kb_info_section:
                    system_content = f"{system_content}\n\n{kb_info_section}"
                
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            
            # Thêm conversation history legacy nếu có (backward compatibility)
            if conversation_history:
                # Chèn history giữa system và user message hiện tại
                messages = [messages[0]] + conversation_history + [messages[1]]
                logger.info(f"[PROMPT_BUILDER] Đã thêm {len(conversation_history)} messages từ conversation history (legacy)")
            
            logger.info(f"[PROMPT_BUILDER] Đã xây dựng {len(messages)} messages với system/user separation")
            return messages
            
        except Exception as e:
            logger.error(f"[PROMPT_BUILDER] Lỗi khi xây dựng messages: {str(e)}", exc_info=True)
            # Fallback to simple format
            return [
                {"role": "system", "content": "Answer based on context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
    
    @staticmethod
    def _messages_to_prompt(messages):
        """Chuyển đổi messages format về prompt string (legacy compatibility)"""
        try:
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            return "\n\n".join(prompt_parts)
        except Exception as e:
            logger.error(f"[PROMPT_BUILDER] Lỗi khi chuyển đổi messages: {str(e)}")
            return "System: Answer based on context.\n\nUser: Please provide your question."

    @staticmethod
    def get_parameters(prompt_builder, request_parameters=None):
        """Lấy các tham số LLM từ cấu hình, có thể ghi đè từ request"""
        params = prompt_builder.get("parameters", {}).copy()

        # Ghi đè các tham số từ request nếu có
        if request_parameters:
            for key, value in request_parameters.items():
                # Chỉ ghi đè các tham số không phải custom_system_instruction và custom_instruction_template
                if key not in ["custom_system_instruction", "custom_instruction_template", "conversation_memory"]:
                    params[key] = value
                    logger.info(f"[PROMPT_BUILDER] Ghi đè tham số {key} từ request")

        # Đảm bảo max_tokens là số nguyên nếu tồn tại
        if "max_tokens" in params and params["max_tokens"] is not None:
            params["max_tokens"] = int(params["max_tokens"])
            
        logger.info(f"[PROMPT_BUILDER] Tham số LLM sau khi xử lý: {json.dumps(params, indent=2, ensure_ascii=False)}")
        return params