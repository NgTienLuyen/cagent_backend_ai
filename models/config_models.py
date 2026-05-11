from pydantic import BaseModel, Field, UUID4
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from enum import Enum


class ModelType(str, Enum):
    ONLINE = "online"
    LOCAL = "local"


class ModelStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"


class LlmConfig(BaseModel):
    model_type: ModelType
    model_name: str
    api_key: Optional[str] = None
    api_keys: Optional[List[str]] = None
    endpoint: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_type: Optional[ModelType] = None  # Thêm embedding_type để phân biệt online/local
    embedding_provider: Optional[str] = None  # Thêm embedding_provider (openai, gemini, cohere, local)
    reranker_model: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class PromptBuilderConfig(BaseModel):
    """
    Cấu hình xây dựng prompt cho LLM

    Các tham số hỗ trợ trong parameters:
    - max_chunks: int - Số lượng chunk tối đa sử dụng (mặc định: 10)
    - use_reranker: bool - Có sử dụng reranking không (mặc định: True)
    - temperature: float - Nhiệt độ cho LLM (mặc định: 0.7)
    - max_tokens: int - Số token tối đa cho câu trả lời (mặc định: 1024)
    - include_kb_description: bool - Có bao gồm mô tả của knowledge base không (mặc định: True)
    """
    system_instruction: str
    context_template: str
    query_template: str
    instruction_template: str
    output_format: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class QueryConfigCreate(BaseModel):
    name_config: str
    knowledge_base_id: UUID4
    llm_config: LlmConfig
    prompt_builder: PromptBuilderConfig
    is_default: bool = False


class QueryConfigResponse(BaseModel):
    id: UUID4
    name_config: str
    knowledgeBaseId: UUID4
    llm_config: LlmConfig
    prompt_builder: PromptBuilderConfig
    is_default: bool
    status: ModelStatus
    create_time: datetime


class QueryConfigUpdate(BaseModel):
    name_config: Optional[str] = None
    llm_config: Optional[LlmConfig] = None
    prompt_builder: Optional[PromptBuilderConfig] = None
    is_default: Optional[bool] = None
    status: Optional[ModelStatus] = None


class QueryRequest(BaseModel):
    query: str
    knowledgeBaseId: UUID4
    config_id: Optional[UUID4] = None
    parameters: Optional[Dict[str, Any]] = None
    chat_section_id: Optional[UUID4] = None


class ChunkResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_link:str
    chunk_text: str
    document_name: str
    similarity_score: float = 0.0
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    retrieved_chunks: List[ChunkResponse]
    llm_answer: str
    model: str
    config_name: str
    config_id: UUID4
    request_id: UUID4
    processing_time: float
