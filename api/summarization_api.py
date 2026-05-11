import asyncio
import logging
import uuid
from fastapi import APIRouter, HTTPException, Depends, Body, Path
from typing import Optional, Dict, Any

from llms.config_loader import ModelConfigLoader
from database.query_config_db import QueryConfigDB
from database.db_summarization import (
    get_document_details_for_summary,
    update_document_description_in_db,
    get_all_document_descriptions_for_kb,
    update_knowledge_base_description_in_db,
    get_knowledge_base_current_description
)
# Cần import hàm read_file_content. Giả sử nó nằm trong api.semantic_chunking_api
# Nếu nó được chuyển đi nơi khác, cần cập nhật đường dẫn import
from api.semantic_chunking_api import read_file_content, ocr_with_tesseract, ocr_with_google_vision
from services.query_service import thread_pool_executor # Tái sử dụng thread_pool_executor

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_llm_instance_for_kb(knowledge_base_id: str):
    """Tải LLM dựa trên cấu hình mặc định của Knowledge Base."""
    kb_config = await QueryConfigDB.get_default_config(knowledge_base_id)
    if not kb_config or "llm_config" not in kb_config:
        logger.error(f"Không tìm thấy cấu hình LLM mặc định cho KB {knowledge_base_id}")
        raise HTTPException(status_code=404, detail=f"Không có cấu hình LLM cho Knowledge Base ID {knowledge_base_id}")
    
    llm_config = kb_config["llm_config"]
    try:
        llm_instance = ModelConfigLoader.load_model(llm_config)
        logger.info(f"Đã tải LLM {llm_config.get('model_name')} cho KB {knowledge_base_id}")
        return llm_instance, llm_config
    except Exception as e:
        logger.error(f"Lỗi khi tải LLM cho KB {knowledge_base_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi tải LLM: {str(e)}")

@router.post("/documents/{document_id}/summarize-description", tags=["Summarization"])
async def summarize_document_description(
    document_id: str = Path(..., description="ID của tài liệu cần tóm tắt mô tả"),
    force_regenerate: bool = Body(False, description="Bắt buộc tạo lại mô tả ngay cả khi đã tồn tại")
):
    """
    Tự động đọc nội dung tài liệu, dùng LLM tóm tắt và cập nhật vào trường `description`.
    """
    logger.info(f"Yêu cầu tóm tắt mô tả cho document_id: {document_id}, force_regenerate: {force_regenerate}")
    loop = asyncio.get_event_loop()

    try:
        details = get_document_details_for_summary(document_id)

        if not details:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tài liệu với ID: {document_id}")
        
        doc_link, kb_id, current_description = details

        if current_description and not force_regenerate:
            logger.info(f"Tài liệu {document_id} đã có mô tả. Bỏ qua tóm tắt.")
            return {"message": "Tài liệu đã có mô tả.", "document_id": document_id, "description": current_description}

        if not doc_link:
            raise HTTPException(status_code=404, detail=f"Tài liệu {document_id} không có đường dẫn (document_link).")

        logger.info(f"Đọc nội dung tài liệu từ: {doc_link}")
        # Giả sử read_file_content có thể chạy trong executor nếu nó là blocking
        # Hoặc nếu nó đã là async, không cần executor ở đây
        file_content = await read_file_content(doc_link, ocr_method="auto") # Sử dụng ocr_method auto
        
        if not file_content.strip():
            logger.warning(f"Nội dung tài liệu {document_id} rỗng.")
            raise HTTPException(status_code=400, detail="Nội dung tài liệu rỗng, không thể tóm tắt.")

        llm, llm_cfg = await get_llm_instance_for_kb(kb_id)

        prompt = f"""
        Bạn là một trợ lý AI chuyên nghiệp.
        Nhiệm vụ của bạn là đọc văn bản sau đây và tạo ra một bản tóm tắt ngắn gọn, súc tích (tối đa 3-4 câu, khoảng 100-150 từ) bằng tiếng Việt, nắm bắt các ý chính và mục đích của tài liệu.
        Không thêm thông tin không có trong văn bản gốc.
        Chỉ trả lời bằng nội dung tóm tắt, không thêm lời chào hay bất kỳ câu dẫn nào khác.
        
        Văn bản cần tóm tắt:
        --- 
        {file_content[:15000]} 
        --- 
        
        Tóm tắt:
        """ # Giới hạn nội dung đầu vào để tránh quá tải LLM

        logger.info(f"Gọi LLM ({llm_cfg.get('model_name')}) để tóm tắt tài liệu {document_id}")
        
        # Chạy LLM call trong executor nếu generate_content là blocking
        llm_response = await loop.run_in_executor(
            thread_pool_executor,
            lambda: llm.generate_content(prompt) # generate_content không có tham số async trực tiếp
        )

        summary = ""
        if isinstance(llm_response, dict) and "content" in llm_response:
            summary = llm_response["content"].strip()
        elif isinstance(llm_response, str):
            summary = llm_response.strip()
        
        if not summary:
            logger.error(f"LLM không trả về bản tóm tắt cho tài liệu {document_id}")
            raise HTTPException(status_code=500, detail="LLM không tạo được bản tóm tắt.")

        logger.info(f"LLM đã tạo tóm tắt cho {document_id}: {summary[:100]}...")

        success = await loop.run_in_executor(thread_pool_executor, update_document_description_in_db, document_id, summary)

        if success:
            logger.info(f"Đã cập nhật thành công mô tả cho tài liệu {document_id}")
            return {"message": "Tóm tắt và cập nhật mô tả tài liệu thành công.", "document_id": document_id, "description": summary}
        else:
            logger.error(f"Không thể cập nhật mô tả cho tài liệu {document_id} vào DB.")
            raise HTTPException(status_code=500, detail="Lỗi khi cập nhật mô tả vào cơ sở dữ liệu.")

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTPException để FastAPI xử lý
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tóm tắt mô tả tài liệu {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ: {str(e)}")

@router.post("/knowledge-bases/{knowledge_base_id}/summarize-description", tags=["Summarization"])
async def summarize_knowledge_base_description(
    knowledge_base_id: str = Path(..., description="ID của Knowledge Base cần tóm tắt mô tả"),
    force_regenerate: bool = Body(False, description="Bắt buộc tạo lại mô tả ngay cả khi đã tồn tại")
):
    """
    Tổng hợp mô tả từ các tài liệu thuộc Knowledge Base, dùng LLM tóm tắt và cập nhật.
    """
    logger.info(f"Yêu cầu tóm tắt mô tả cho knowledge_base_id: {knowledge_base_id}, force_regenerate: {force_regenerate}")
    loop = asyncio.get_event_loop()

    try:
        current_kb_description = await loop.run_in_executor(thread_pool_executor, get_knowledge_base_current_description, knowledge_base_id)

        if current_kb_description and not force_regenerate:
            logger.info(f"Knowledge Base {knowledge_base_id} đã có mô tả. Bỏ qua tóm tắt.")
            return {"message": "Knowledge Base đã có mô tả.", "knowledge_base_id": knowledge_base_id, "description": current_kb_description}

        doc_descriptions = await loop.run_in_executor(thread_pool_executor, get_all_document_descriptions_for_kb, knowledge_base_id)

        if not doc_descriptions:
            logger.warning(f"Không tìm thấy mô tả tài liệu nào cho Knowledge Base {knowledge_base_id} để tổng hợp.")
            raise HTTPException(status_code=404, detail="Không có mô tả tài liệu nào để tổng hợp cho Knowledge Base này.")

        combined_descriptions = "\n\n---\n\n".join(doc_descriptions)
        logger.info(f"Đã tổng hợp {len(doc_descriptions)} mô tả tài liệu cho KB {knowledge_base_id}. Tổng độ dài: {len(combined_descriptions)} ký tự.")

        llm, llm_cfg = await get_llm_instance_for_kb(knowledge_base_id)

        prompt = f"""
        Bạn là một chuyên gia biên tập nội dung AI.
        Dưới đây là một tập hợp các mô tả ngắn từ nhiều tài liệu khác nhau trong cùng một cơ sở kiến thức. 
        Nhiệm vụ của bạn là đọc tất cả các mô tả này và viết một mô tả tổng hợp, chuyên nghiệp (khoảng 150-250 từ) cho toàn bộ cơ sở kiến thức. 
        Mô tả tổng hợp này cần làm nổi bật các chủ đề chính, phạm vi kiến thức và mục đích chung của cơ sở kiến thức dựa trên các tài liệu thành phần.
        Hãy trình bày một cách mạch lạc, dễ hiểu và bao quát.
        Chỉ trả lời bằng nội dung mô tả tổng hợp, không thêm lời chào hay bất kỳ câu dẫn nào khác.
        
        Các mô tả tài liệu:
        --- 
        {combined_descriptions[:20000]} 
        --- 
        
        Mô tả tổng hợp cho Cơ sở kiến thức:
        """ # Giới hạn nội dung đầu vào

        logger.info(f"Gọi LLM ({llm_cfg.get('model_name')}) để tóm tắt mô tả cho KB {knowledge_base_id}")
        
        llm_response = await loop.run_in_executor(
            thread_pool_executor,
            lambda: llm.generate_content(prompt)
        )

        summary = ""
        if isinstance(llm_response, dict) and "content" in llm_response:
            summary = llm_response["content"].strip()
        elif isinstance(llm_response, str):
            summary = llm_response.strip()

        if not summary:
            logger.error(f"LLM không trả về bản tóm tắt cho KB {knowledge_base_id}")
            raise HTTPException(status_code=500, detail="LLM không tạo được bản tóm tắt tổng hợp.")

        logger.info(f"LLM đã tạo tóm tắt cho KB {knowledge_base_id}: {summary[:100]}...")

        success = await loop.run_in_executor(thread_pool_executor, update_knowledge_base_description_in_db, knowledge_base_id, summary)

        if success:
            logger.info(f"Đã cập nhật thành công mô tả cho Knowledge Base {knowledge_base_id}")
            return {"message": "Tóm tắt và cập nhật mô tả Knowledge Base thành công.", "knowledge_base_id": knowledge_base_id, "description": summary}
        else:
            logger.error(f"Không thể cập nhật mô tả cho Knowledge Base {knowledge_base_id} vào DB.")
            raise HTTPException(status_code=500, detail="Lỗi khi cập nhật mô tả Knowledge Base vào cơ sở dữ liệu.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tóm tắt mô tả Knowledge Base {knowledge_base_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi tóm tắt KB: {str(e)}")

# Để router này được FastAPI nhận diện, bạn cần import nó vào main.py
# Ví dụ trong main.py: from api.summarization_api import router as summarization_router
# Và app.include_router(summarization_router, prefix="/api/v1") 