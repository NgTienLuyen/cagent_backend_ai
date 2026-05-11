import pytest
from llms.localLllms import LocalLlms  # Import class LLM


@pytest.fixture
def llm():
    """Khởi tạo LLM để dùng cho các test case."""
    return LocalLlms(model_name="deepseek-r1:1.5b")  # Hoặc "llama2:latest"


def test_chat_response(llm):
    """Kiểm tra API chat của LLM có trả về dữ liệu không."""
    messages = [{"role": "user", "content": "LangChain là gì?"}]
    response = llm.chat(messages)

    assert isinstance(response, str)  # Phải trả về kiểu string
    assert len(response) > 0  # Không được rỗng
    print(f"\n🔹 Phản hồi từ LLM: {response}")


def test_generate_content(llm):
    """Kiểm tra API generate_content của LLM có hoạt động không."""
    prompt = "Hãy mô tả về LangChain."
    response = llm.generate_content(prompt)

    assert isinstance(response, str)  # Phải trả về kiểu string
    assert len(response) > 0  # Không được rỗng
    print(f"\n🔹 Nội dung sinh ra: {response}")
