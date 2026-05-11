import os
import backoff
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
# Sửa import Cohere
try:
    # Thử import từ package mới
    from langchain_cohere import CohereEmbeddings
    from langchain_cohere import ChatCohere
except ImportError:
    # Fallback sang package cũ với thêm user_agent
    from langchain.embeddings import CohereEmbeddings
    try:
        from langchain.chat_models import ChatCohere
    except ImportError:
        # Nếu không có ChatCohere, sử dụng một class giả
        class ChatCohere:
            def __init__(self, *args, **kwargs):
                raise ImportError("langchain_cohere package không được cài đặt")
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .base import LLM
import time
import requests

# Load API Keys từ file .env (fallback nếu không được cung cấp trong params)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

logger = logging.getLogger(__name__)

# Danh sách các mô hình tham khảo - chỉ để tham khảo, không giới hạn model được sử dụng
AVAILABLE_MODELS = {
    "GPT-4": "gpt-4-turbo-preview",
    "GPT-4o": "gpt-4o",
    "GPT-3.5": "gpt-3.5-turbo",
    "GPT-3.5-turbo": "gpt-3.5-turbo",
    "Gemini Pro": "gemini-pro",
    "Cohere Command R": "command-r",
    # Có thể thêm các mô hình khác
}

# Danh sách các embedding model tương ứng - chỉ để tham khảo
EMBEDDING_MODELS = {
    "GPT-4": "text-embedding-3-large",
    "GPT-4o": "text-embedding-3-large",
    "GPT-3.5": "text-embedding-3-small",
    "Gemini Pro": "models/embedding-001",
    "Cohere Command R": "embed-english-v3.0",
    # Có thể thêm các mô hình khác
}

# Provider identification for model_name
PROVIDER_HINTS = {
    "openai": ["gpt", "text-davinci", "davinci", "text-embedding-3"],
    "gemini": ["gemini", "text-embedding-004", "embedding-001"],
    "cohere": ["cohere", "command", "embed-", "embed-multilingual", "command-a"],
    "anthropic": ["claude", "anthropic"],
    "mistral": ["mistral"],
    "huggingface": ["hf"],
    "azure": ["azure"],
    "anyscale": ["anyscale"],
    "langchain": ["langchain"],
}


class OnlineLLMs(LLM):
    def __init__(self, model_name: str = "GPT-4", api_key: Optional[str] = None, endpoint: Optional[str] = None, provider: Optional[str] = None) -> None:
        super().__init__(model_name)
        self.model_type = "online"
        self.api_key = api_key  # API key từ cấu hình
        self.endpoint = endpoint  # Custom endpoint nếu có
        
        logger.info(f"Initializing Online LLM with model: {model_name}")
        
        # Sử dụng provider được cung cấp hoặc phát hiện từ tên model
        # Chuẩn hóa provider: map 'google' -> 'gemini'
        if provider and provider.lower() == "google":
            provider = "gemini"
        self.provider = provider if provider else self._detect_provider(model_name)
        logger.info(f"Using provider: {self.provider}")
        
        self.client = self._initialize_client(model_name)
        # Expose api_key for rotation/cooldown tracking
        self.api_key = self.api_key
        self.embedding_client = self._initialize_embedding_client(model_name)
        logger.info(f"Successfully initialized Online LLM: {model_name}")
        
        # Track query timings
        self.timings = {"last_query_time": 0, "total_tokens": 0}

    def _detect_provider(self, model_name: str) -> str:
        """Phát hiện nhà cung cấp dựa trên tên model"""
        model_name_lower = model_name.lower()
        
        for provider, hints in PROVIDER_HINTS.items():
            for hint in hints:
                if hint in model_name_lower:
                    return provider
                    
        # Default fallback to OpenAI if unknown
        return "openai"

    def _initialize_client(self, model_name: str) -> Any:
        """Khởi tạo API LangChain dựa trên mô hình"""
        try:
            # Xác định model_id từ AVAILABLE_MODELS nếu có, nếu không dùng model_name
            model_id = AVAILABLE_MODELS.get(model_name, model_name)
            logger.info(f"Using model ID: {model_id}")
            
            # Sử dụng provider đã được xác định trong constructor
            provider = self.provider
            
            # Lấy API key, ưu tiên từ constructor, sau đó từ biến môi trường
            if provider == "openai":
                api_key = self.api_key or OPENAI_API_KEY
                if not api_key:
                    raise ValueError("Missing OpenAI API Key")
                    
                # Sử dụng custom endpoint nếu được cung cấp
                if self.endpoint:
                    return ChatOpenAI(model_name=model_id, openai_api_key=api_key, openai_api_base=self.endpoint)
                return ChatOpenAI(model_name=model_id, openai_api_key=api_key)
                
            elif provider == "gemini":
                api_key = self.api_key or GEMINI_API_KEY
                if not api_key:
                    raise ValueError("Missing Gemini API Key")
                return ChatGoogleGenerativeAI(model=model_id, google_api_key=api_key)
                
            elif provider == "cohere":
                api_key = self.api_key or COHERE_API_KEY
                if not api_key:
                    raise ValueError("Missing Cohere API Key")
                # Sử dụng ChatCohere thay vì ChatOpenAI
                try:
                    return ChatCohere(model=model_id, cohere_api_key=api_key)
                except (ImportError, TypeError) as e:
                    logger.error(f"Không thể khởi tạo ChatCohere: {str(e)}")
                    raise ValueError(f"Lỗi khi khởi tạo ChatCohere: {str(e)}. Hãy cài đặt langchain-cohere: pip install langchain-cohere")
                
            elif provider == "anthropic":
                if not self.api_key:
                    raise ValueError("Missing Anthropic API Key")
                # Sử dụng Anthropic thông qua LangChain
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=model_id, anthropic_api_key=self.api_key)
                
            elif provider == "mistral":
                if not self.api_key:
                    raise ValueError("Missing Mistral AI API Key")
                # Tạo client Mistral thông qua LangChain hoặc API trực tiếp
                from langchain_mistralai.chat_models import ChatMistralAI
                return ChatMistralAI(model=model_id, mistralai_api_key=self.api_key)
                
            else:
                # Mặc định sử dụng OpenAI (hoặc endpoint tùy chỉnh) nếu không nhận diện được
                api_key = self.api_key or OPENAI_API_KEY
                if not api_key:
                    raise ValueError(f"Missing API Key for provider: {provider}")
                    
                # Nếu có endpoint tùy chỉnh, sử dụng endpoint đó
                if self.endpoint:
                    return ChatOpenAI(model_name=model_id, openai_api_key=api_key, openai_api_base=self.endpoint)
                    
                logger.warning(f"Unrecognized model provider: {provider}. Using OpenAI client by default.")
                return ChatOpenAI(model_name=model_id, openai_api_key=api_key)
                
        except Exception as e:
            logger.error(f"Failed to initialize client for {model_name}: {str(e)}")
            raise

    def _initialize_embedding_client(self, model_name: str) -> Any:
        """Khởi tạo client cho embedding dựa trên loại mô hình"""
        try:
            # Sử dụng provider đã được xác định trong constructor
            provider = self.provider
            
            # Nếu không có embedding model, sử dụng model tương ứng với provider
            if provider == "openai":
                embedding_model = EMBEDDING_MODELS.get(model_name, "text-embedding-3-small")
                api_key = self.api_key or OPENAI_API_KEY
                if not api_key:
                    raise ValueError("Missing OpenAI API Key")
                # Sử dụng custom endpoint nếu được cung cấp
                if self.endpoint:
                    return OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key, openai_api_base=self.endpoint)
                return OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key)
                
            elif provider == "gemini":
                embedding_model = EMBEDDING_MODELS.get("Gemini Pro", "models/embedding-001")
                api_key = self.api_key or GEMINI_API_KEY
                if not api_key:
                    raise ValueError("Missing Gemini API Key")
                return GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key)
                
            elif provider == "cohere":
                embedding_model = EMBEDDING_MODELS.get("Cohere Command R", "embed-english-v3.0")
                api_key = self.api_key or COHERE_API_KEY
                if not api_key:
                    raise ValueError("Missing Cohere API Key")
                try:
                    # Thử khởi tạo với package mới
                    return CohereEmbeddings(model=embedding_model, cohere_api_key=api_key)
                except TypeError:
                    # Fallback: Thêm user_agent nếu package cũ
                    return CohereEmbeddings(
                        model=embedding_model, 
                        cohere_api_key=api_key,
                        user_agent="langchain"
                    )
                
            else:
                # Mặc định sử dụng OpenAI embedding
                embedding_model = EMBEDDING_MODELS.get(model_name, "text-embedding-3-small")
                api_key = self.api_key or OPENAI_API_KEY
                if not api_key:
                    raise ValueError("Missing API Key for embedding")
                    
                # Sử dụng custom endpoint nếu được cung cấp
                if self.endpoint:
                    return OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key, openai_api_base=self.endpoint)
                    
                logger.warning(f"Unrecognized provider for embedding: {provider}. Using OpenAI embedding by default.")
                return OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key)
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding client for {model_name} with provider {self.provider}: {str(e)}")
            raise

    def _messages_to_string(self, messages: List[Dict[str, str]]) -> str:
        """Chuyển đổi messages format về string format"""
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
            logger.error(f"[ONLINE_LLM] Lỗi khi chuyển đổi messages: {str(e)}")
            return "System: Answer based on context.\n\nUser: Please provide your question."

    def generate_content_with_messages(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate content using messages format (tối ưu hơn)
        
        Args:
            messages: List[Dict] - Messages format [{"role": "system", "content": "..."}, ...]
            temperature: float - Temperature for generation
            max_tokens: int - Maximum tokens to generate
        
        Returns:
            str: Generated content
        """
        # Đảm bảo max_tokens là số nguyên
        max_tokens = int(max_tokens) if max_tokens is not None else 1024
        temperature = float(temperature) if temperature is not None else 0.7

        logger.info(f"[ONLINE_LLM] Generating content with messages format using {self.model_name} (temp: {temperature}, max_tokens: {max_tokens})")
        
        # Chuyển đổi messages format sang langchain format
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        try:
            # Xử lý riêng biệt cho từng provider
            if self.provider == "gemini":
                response = self.llm.invoke(langchain_messages, temperature=temperature, max_output_tokens=max_tokens)
            elif self.provider == "openai":
                response = self.llm.invoke(langchain_messages, temperature=temperature, max_tokens=max_tokens)
            elif self.provider == "cohere":
                response = self.llm.invoke(langchain_messages, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating content with messages: {str(e)}")
            # Mark API key as rate limited if it's a rate limit error
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                from .config_loader import mark_rate_limited
                if self.api_key:
                    mark_rate_limited(self.api_key, seconds=120)  # 2 phút cooldown
            raise

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Tạo nội dung từ prompt sử dụng online LLM.
        
        Args:
            prompt: Văn bản prompt đầu vào
            temperature: Độ sáng tạo của model (0.0-1.0)
            max_tokens: Số token tối đa trong response
            
        Returns:
            str: Nội dung được tạo ra
            
        Raises:
            Exception: Khi có lỗi trong quá trình gọi API
        """
        # Đảm bảo max_tokens là số nguyên
        max_tokens = int(max_tokens) if max_tokens is not None else 1024
        temperature = float(temperature) if temperature is not None else 0.7

        logger.info(f"Generating content with {self.model_name} (temp: {temperature}, max_tokens: {max_tokens})")
        
        # Nếu prompt là messages format, chuyển đổi về string
        if isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg for msg in prompt):
            logger.info(f"[ONLINE_LLM] Detected messages format, converting to string")
            prompt = self._messages_to_string(prompt)
        messages = [HumanMessage(content=prompt)]

        try:
            # Xử lý riêng biệt cho từng provider
            if self.provider == "gemini":
                logger.info(f"Using specialized params for Gemini models")
                # Gemini 2.0-flash không hỗ trợ các tham số configuration 
                if "flash" in self.model_name.lower():
                    logger.info(f"Flash model detected, calling without parameters")
                    # Đối với Gemini Flash, chỉ cần gọi không cần thêm tham số
                    # Thêm timeout để tránh treo (cross-platform solution)
                    import concurrent.futures
                    import threading
                    
                    def call_gemini():
                        return self.client.invoke(messages)
                    
                    # Set timeout 30 giây
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(call_gemini)
                        try:
                            response = future.result(timeout=30)
                        except concurrent.futures.TimeoutError:
                            logger.error("Gemini Flash API call timed out after 30 seconds")
                            raise Exception("Gemini API call timed out. Please try again.")
                        except Exception as e:
                            logger.error(f"Gemini Flash API call failed: {str(e)}")
                            raise
                else:
                    # Thử với model Gemini thông thường, có thể cần đơn giản hóa tham số hơn nữa
                    logger.info(f"Non-flash Gemini model, using simplified parameters")
                    try:
                        # Thử với temperature nhưng không có max_output_tokens
                        response = self.client.invoke(
                            messages,
                            temperature=temperature
                        )
                    except Exception as e:
                        logger.warning(f"Failed with temperature, trying without parameters: {str(e)}")
                        # Nếu lỗi, thử gọi không tham số
                        response = self.client.invoke(messages)
            else:
                # Các provider khác như OpenAI, Anthropic, v.v.
                response = self.client.invoke(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            return response.content.strip() if response.content else "⚠️ Không có phản hồi từ LLM."
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            # Kiểm tra nếu là lỗi rate limit/quota exceeded
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['429', 'quota', 'rate limit', 'exceeded']):
                logger.warning(f"Rate limit detected for API key, marking for cooldown")
                # Import mark_rate_limited function
                from .config_loader import mark_rate_limited
                if self.api_key:
                    mark_rate_limited(self.api_key, seconds=900)  # 15 phút cooldown
            raise

    def generate_content_with_timing(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """Generate content and track timing information"""
        start_time = time.time()
        
        content = self.generate_content(prompt, temperature, max_tokens)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.timings["last_query_time"] = processing_time
        # Approximate token count - in a real implementation, you'd use a tokenizer
        self.timings["total_tokens"] += len(prompt.split()) / 4  
        
        return {
            "content": content,
            "processing_time": processing_time
        }

    async def generate_content_stream_with_messages(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024):
        """
        Generate content stream using messages format (tối ưu hơn)
        
        Args:
            messages: List[Dict] - Messages format [{"role": "system", "content": "..."}, ...]
            temperature: float - Temperature for generation
            max_tokens: int - Maximum tokens to generate
        
        Yields:
            str: Streaming tokens
        """
        # Đảm bảo max_tokens là số nguyên
        max_tokens = int(max_tokens) if max_tokens is not None else 1024
        temperature = float(temperature) if temperature is not None else 0.7

        logger.info(f"[ONLINE_LLM] Streaming content with messages format using {self.model_name} (temp: {temperature}, max_tokens: {max_tokens})")
        
        # Chuyển đổi messages format sang langchain format
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        try:
            # Xử lý riêng biệt cho từng provider
            if self.provider == "openai":
                # OpenAI streaming với messages
                async for chunk in self.client.astream(
                    langchain_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
                        
            elif self.provider == "gemini":
                # Gemini streaming với messages
                if "flash" in self.model_name.lower():
                    logger.info(f"Streaming Gemini Flash model with messages: {self.model_name}")
                    async for chunk in self.client.astream(langchain_messages):
                        if hasattr(chunk, 'content') and chunk.content:
                            yield chunk.content
                else:
                    logger.info(f"Streaming Gemini regular model with messages: {self.model_name}")
                    try:
                        async for chunk in self.client.astream(
                            langchain_messages,
                            temperature=temperature
                        ):
                            if hasattr(chunk, 'content') and chunk.content:
                                yield chunk.content
                    except Exception as e:
                        logger.warning(f"Gemini streaming with temperature failed: {str(e)}, trying without parameters")
                        async for chunk in self.client.astream(langchain_messages):
                            if hasattr(chunk, 'content') and chunk.content:
                                yield chunk.content
                                
            elif self.provider == "cohere":
                # Cohere streaming với messages
                async for chunk in self.client.astream(
                    langchain_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
                        
        except Exception as e:
            logger.error(f"Streaming with messages failed: {str(e)}")
            # Fallback to non-streaming
            response = self.generate_content_with_messages(messages, temperature, max_tokens)
            yield response

    async def generate_content_stream(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024):
        """
        Generate content with streaming support for different providers.
        Returns an async generator that yields tokens.
        """
        from llms.stream_handlers import StreamingCallbackHandler
        
        # Đảm bảo max_tokens là số nguyên
        max_tokens = int(max_tokens) if max_tokens is not None else 1024
        temperature = float(temperature) if temperature is not None else 0.7

        logger.info(f"Streaming content with {self.model_name} (provider: {self.provider}, temp: {temperature}, max_tokens: {max_tokens})")
        
        # Tạo streaming callback handler
        handler = StreamingCallbackHandler()
        
        # Import langchain streaming components
        from langchain.schema import HumanMessage
        
        try:
            if self.provider == "openai":
                # OpenAI streaming với LangChain
                messages = [HumanMessage(content=prompt)]
                
                # Clone client với streaming callback
                streaming_client = self.client.copy()
                streaming_client.streaming = True
                streaming_client.callbacks = [handler]
                
                # Start streaming generation
                import asyncio
                
                async def _stream_openai():
                    try:
                        # Gọi invoke với callback để trigger streaming
                        response = streaming_client.invoke(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # Chờ stream kết thúc
                        while True:
                            token = await handler.queue.get()
                            if token is None:
                                break
                            yield token
                            
                    except Exception as e:
                        logger.error(f"OpenAI streaming error: {str(e)}")
                        await handler.queue.put(None)
                        
                async for token in _stream_openai():
                    yield token
                    
            elif self.provider == "gemini":
                # Gemini streaming implementation
                messages = [HumanMessage(content=prompt)]
                
                async def _stream_gemini():
                    try:
                        # Xử lý riêng biệt cho từng loại Gemini model
                        if "flash" in self.model_name.lower():
                            # Gemini Flash models - no parameters
                            logger.info(f"Streaming Gemini Flash model: {self.model_name}")
                            # Sử dụng astream để streaming với Gemini
                            async for chunk in self.client.astream(messages):
                                if hasattr(chunk, 'content') and chunk.content:
                                    yield chunk.content
                        else:
                            # Gemini regular models
                            logger.info(f"Streaming Gemini regular model: {self.model_name}")
                            try:
                                # Thử với temperature
                                async for chunk in self.client.astream(
                                    messages,
                                    temperature=temperature
                                ):
                                    if hasattr(chunk, 'content') and chunk.content:
                                        yield chunk.content
                            except Exception as e:
                                logger.warning(f"Gemini streaming with temperature failed: {str(e)}, trying without parameters")
                                # Fallback: no parameters
                                async for chunk in self.client.astream(messages):
                                    if hasattr(chunk, 'content') and chunk.content:
                                        yield chunk.content
                                        
                    except Exception as e:
                        logger.error(f"Gemini streaming error: {str(e)}")
                        # Fallback to non-streaming if streaming fails
                        response = self.generate_content(prompt, temperature, max_tokens)
                        yield response
                        
                async for token in _stream_gemini():
                    yield token
                    
            elif self.provider == "cohere":
                # Cohere streaming - tương tự pattern
                messages = [HumanMessage(content=prompt)]
                
                async def _stream_cohere():
                    try:
                        async for chunk in self.client.astream(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ):
                            if hasattr(chunk, 'content') and chunk.content:
                                yield chunk.content
                    except Exception as e:
                        logger.error(f"Cohere streaming error: {str(e)}")
                        # Fallback to non-streaming
                        response = self.generate_content(prompt, temperature, max_tokens)
                        yield response
                        
                async for token in _stream_cohere():
                    yield token
                    
            else:
                # Fallback cho providers khác - non-streaming
                logger.warning(f"Streaming not implemented for provider: {self.provider}, fallback to non-streaming")
                response = self.generate_content(prompt, temperature, max_tokens)
                yield response
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            # Fallback to non-streaming
            response = self.generate_content(prompt, temperature, max_tokens)
            yield response

    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1):
        """Tạo thông điệp dựa trên prompt hệ thống"""
        logger.info(f"Creating agentic chunker message with {self.model_name}")
        system_msg = SystemMessage(content=system_prompt)

        user_msgs = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                user_msgs.append(HumanMessage(content=msg["content"]))
            else:
                user_msgs.append(HumanMessage(content=str(msg)))

        formatted_messages = [system_msg] + user_msgs

        try:
            response = self.client(formatted_messages).content
            return response.strip() if response else "⚠️ Không có phản hồi từ LLM."
        except Exception as e:
            logger.error(f"Error in agentic chunker: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Tạo embedding từ văn bản"""
        logger.info(f"Generating embedding with {self.model_name} (provider: {self.provider})")
        try:
            return self.embedding_client.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Kiểm tra nếu là lỗi rate limit/quota exceeded
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['429', 'quota', 'rate limit', 'exceeded']):
                logger.warning(f"Rate limit detected for API key, marking for cooldown")
                # Import mark_rate_limited function
                from .config_loader import mark_rate_limited
                if self.api_key:
                    mark_rate_limited(self.api_key, seconds=900)  # 15 phút cooldown
            raise