# cohere_llm.py
import cohere
from .base import LLM  # Đảm bảo lớp cơ sở LLM được nhập đúng

class CohereLLM(LLM):
    def __init__(self, api_key: str):
        """
        Khởi tạo CohereLLM với API key của Cohere
        """
        self.client = cohere.Client(api_key)  # Khởi tạo client Cohere với API key

    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=200, temperature=0.2):
        """
        Tạo tin nhắn gửi đến Cohere LLM để phân tích và trả về các chunk.
        """
        response = self.client.chat(
            model="command-r-plus-08-2024",  # Chọn mô hình của Cohere
            messages=messages,  # Truyền các tin nhắn đến API
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response['message']['content']  # Trả về nội dung tin nhắn từ Cohere

    def generate_content(self, prompt: str):
        """
        Tạo nội dung cho prompt đã cho bằng API Cohere
        """
        response = self.client.generate(
            model="command-r-plus-08-2024",  # Chọn mô hình Cohere
            prompt=prompt,
            max_tokens=1000,  # Số token tối đa trong kết quả
            temperature=0.7
        )

        return response['text']  # Trả về nội dung tạo ra
