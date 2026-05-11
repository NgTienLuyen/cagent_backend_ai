import os
import requests
import backoff
import logging
from openai import OpenAI
from .base import LLM
from typing import List, Dict, Any, Optional
import time

# Endpoint LM Studio (cổng mặc định 1234) - sẽ được ghi đè bởi endpoint trong cấu hình
LM_STUDIO_ENDPOINT = os.getenv("LM_STUDIO_ENDPOINT", "http://127.0.0.1:1234/v1")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # API Key mặc định của LM Studio

# Danh sách các mô hình tham khảo - chỉ để tham khảo, không giới hạn model được sử dụng
AVAILABLE_MODELS = {
    "Vistral 7B Chat": "vistral-7b-chat",
    # Có thể mở rộng danh sách này
}

logger = logging.getLogger(__name__)


class LocalLlms(LLM):
    def __init__(self, model_name: str = "vistral-7b-chat", endpoint: Optional[str] = None) -> None:
        super().__init__(model_name)
        self.model_type = "local"

        # Bỏ việc validate model_name với AVAILABLE_MODELS
        # Sử dụng tên model được cung cấp trực tiếp
        model_id = AVAILABLE_MODELS.get(model_name, model_name)
        logger.info(f"Initializing local LLM with model: {model_id}")
        self.model_name = model_id
        
        # Sử dụng endpoint tùy chỉnh nếu được cung cấp và chuẩn hóa để luôn có /v1
        raw_endpoint = endpoint or LM_STUDIO_ENDPOINT
        if raw_endpoint.endswith("/"):
            raw_endpoint = raw_endpoint[:-1]
        if not raw_endpoint.endswith("/v1"):
            raw_endpoint = f"{raw_endpoint}/v1"
        self.endpoint = raw_endpoint
        logger.info(f"Using endpoint: {self.endpoint}")

        try:
            self.client = OpenAI(base_url=self.endpoint, api_key=API_KEY)
            logger.info(f"Successfully initialized Local LLM: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Local LLM: {str(e)}")
            raise ConnectionError(f"Không thể kết nối tới LLM local endpoint: {str(e)}")
            
        # Track query timings
        self.timings = {"last_query_time": 0, "total_tokens": 0}

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Gửi yêu cầu hội thoại đến mô hình LM Studio"""
        try:
            logger.debug(f"Sending chat request to Local LLM with {len(messages)} messages")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Hỗ trợ cả hai schema: OpenAI (message.content) và một số server (choices[i].text)
            choice = None
            try:
                # OpenAI-compatible
                choice = response.choices[0].message.content
            except Exception:
                try:
                    # Some local servers return `.text` instead
                    choice = response.choices[0].text
                except Exception:
                    choice = None

            if not choice or not isinstance(choice, str):
                # Raise để tầng gọi trên fallback về câu hỏi gốc (đã có try/except)
                raise ValueError(f"Invalid completion response schema: {repr(response)[:200]}")

            return choice.strip()
        except Exception as e:
            logger.error(f"Error in Local LLM API call: {str(e)}", exc_info=True)
            # Ném lỗi để caller xử lý fallback thay vì trả về chuỗi lỗi làm hỏng pipeline
            raise

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Tạo nội dung từ prompt sử dụng local LLM.
        
        Args:
            prompt: Văn bản prompt đầu vào
            temperature: Độ sáng tạo của model (0.0-1.0)
            max_tokens: Số token tối đa trong response
            
        Returns:
            str: Nội dung được tạo ra
            
        Raises:
            ConnectionError: Khi không thể kết nối tới local LLM
            Exception: Khi có lỗi trong quá trình gọi API
        """
        """Sinh nội dung từ mô hình LM Studio với retry logic"""
        logger.info(f"Generating content with {self.model_name}")
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        
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

    def create_agentic_chunker_message(self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 1.0) -> str:
        """Tạo thông điệp dựa trên prompt hệ thống"""
        logger.info(f"Creating agentic chunker message with {self.model_name}")
        formatted_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})

        return self.chat(formatted_messages, temperature=temperature, max_tokens=max_tokens)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for text - fallback implementation for local models
        """
        logger.warning("Local LLM doesn't support native embeddings. Using fallback method.")
        # Fallback to a simple embedding method or raise an error
        import numpy as np
        import hashlib

        # Generate a deterministic pseudo-embedding based on text hash
        # This is NOT suitable for production use!
        hash_object = hashlib.md5(text.encode())
        seed = int(hash_object.hexdigest(), 16) % (10 ** 8)
        np.random.seed(seed)

        # Generate a 384-dimensional embedding (common dimension)
        return np.random.normal(0, 0.1, 384).tolist()

# import os
# import requests
# import subprocess
# from dotenv import load_dotenv
# from .base import LLM
# import backoff
#
# # Load environment variables
# load_dotenv()
#
# # Ollama API Endpoint
# OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
#
# # Danh sách các mô hình có sẵn
# AVAILABLE_MODELS = {
#     "DeepSeek R1 1.5B": "deepseek-r1:1.5b",
#     "Llama 2": "llama2:latest",
# }
#
#
# class LocalLlms(LLM):
#     def __init__(self, model_name="deepseek-r1:1.5b"):
#         if model_name not in AVAILABLE_MODELS.values():
#             raise ValueError(f"❌ Model không hợp lệ! Chỉ hỗ trợ: {list(AVAILABLE_MODELS.keys())}")
#
#         self.model_name = model_name
#         self.base_url = OLLAMA_ENDPOINT
#         self.ensure_ollama_running()
#         self.pull_model()
#
#     def ensure_ollama_running(self):
#         """Kiểm tra Ollama có đang chạy không, nếu chưa thì khởi động."""
#         try:
#             response = requests.get(f"{self.base_url}/api/tags", timeout=2)
#             if response.status_code == 200:
#                 print("✅ Ollama đã chạy.")
#                 return
#         except requests.ConnectionError:
#             print("⚠️ Ollama chưa chạy, đang khởi động...")
#
#         self.run_ollama()
#
#     def run_ollama(self):
#         """Khởi động Ollama server nếu chưa chạy."""
#         try:
#             subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print("✅ Ollama đã khởi động thành công!")
#         except Exception as e:
#             raise RuntimeError(f"❌ Lỗi khi khởi động Ollama: {str(e)}")
#
#     def pull_model(self):
#         """Tải mô hình từ Ollama nếu chưa có."""
#         response = requests.post(f"{self.base_url}/api/pull", json={"model": self.model_name})
#         if response.status_code != 200:
#             raise Exception(f"❌ Không thể tải model {self.model_name}: {response.text}")
#
#     def chat(self, messages, temperature=0.7, max_tokens=1024):
#         """Gửi tin nhắn đến mô hình LLM và nhận phản hồi."""
#         data = {
#             "model": self.model_name,
#             "messages": messages,
#             "stream": False,
#             "options": {"temperature": temperature, "num_ctx": max_tokens}
#         }
#         response = requests.post(f"{self.base_url}/api/chat", json=data)
#
#         if response.status_code == 200:
#             return response.json().get("message", {}).get("content", "⚠️ Không có phản hồi từ LLM.")
#         else:
#             return f"⚠️ Lỗi API: {response.status_code} - {response.text}"
#
#     @backoff.on_exception(backoff.expo, Exception, max_tries=3)
#     def generate_content(self, prompt):
#         """Tạo nội dung dựa trên prompt."""
#         data = {"model": self.model_name, "prompt": prompt, "stream": False}
#         response = requests.post(f"{self.base_url}/api/generate", json=data)
#
#         if response.status_code == 200:
#             return response.json().get("response", "⚠️ Không có phản hồi từ LLM.")
#         else:
#             return f"⚠️ Lỗi API: {response.status_code} - {response.text}"
#
#     def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1):
#         """Tạo tin nhắn chunker dựa trên LLM."""
#         try:
#             ollama_messages = [{"role": "system", "content": system_prompt}] + messages
#             response = self.chat(ollama_messages, temperature=temperature, max_tokens=max_tokens)
#             return response
#         except Exception as e:
#             print(f"⚠️ Lỗi khi tạo chunker message: {e}")
#             return ""
