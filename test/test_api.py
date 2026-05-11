import pytest
from chunking.llm_agentic_chunker import LLMAgenticChunkerv2
from llms.cohere_llm import CohereLLM
from constant import COHERE  # Đảm bảo bạn đã có constant COHERE được định nghĩa

# Giả sử bạn đã có một API key Cohere hợp lệ
API_KEY = '9lRawWOiQK35jXEwzXG9Pe1QdC0umabHh6kK0cu4'

@pytest.fixture
def cohere_llm():
    """Fixture to initialize Cohere LLM."""
    return CohereLLM(api_key=API_KEY)

@pytest.fixture
def chunker(cohere_llm):
    """Fixture to initialize the chunker."""
    return LLMAgenticChunkerv2(cohere_llm)

