import os
import pytest
from llms.cohere_llm import CohereLLM
from chunking.llm_agentic_chunker import LLMAgenticChunkerv2

# Đọc API Key từ biến môi trường
API_KEY = os.getenv("COHERE_API_KEY", "9lRawWOiQK35jXEwzXG9Pe1QdC0umabHh6kK0cu4")  # Thay thế bằng API Key thực tế

@pytest.fixture
def chunker():
    cohere_llm = CohereLLM(API_KEY)
    return LLMAgenticChunkerv2(cohere_llm)

def test_chunking(chunker):
    text = """ Machine learning is a subset of AI. It consists of supervised and unsupervised learning techniques.Supervised learning relies on labeled data, while unsupervised learning finds patterns in data.Reinforcement learning is a different approach where an agent learns through rewards and penalties."""

    chunks = chunker.split_text(text)

    # Kiểm tra có ít nhất một chunk
    assert len(chunks) > 0, "Chunking không hoạt động, không có chunk nào được tạo."

    # In kết quả (tuỳ chọn)
    print("\n Kết quả chunking:")
    for idx, chunk in enumerate(chunks, 1):
        print(f"Chunk {idx}: {chunk}")


