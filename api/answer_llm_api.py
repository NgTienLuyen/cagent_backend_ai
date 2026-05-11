# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import List
# from llms.localLllms import LocalLlms  # Gọi LLM cục bộ
#
# router = APIRouter()
#
# # Định nghĩa request model
# class AnswerRequest(BaseModel):
#     question: str  # Câu hỏi của người dùng
#     retrieved_chunks: List[str]  # Danh sách các đoạn văn bản từ API tìm kiếm vector
#     model_name: str  # Chọn mô hình LLM (DeepSeek, Gemini, Cohere,...)
#
# @router.post("/generate_answer")
# async def generate_answer(request: AnswerRequest):
#     """
#     API nhận câu hỏi và các đoạn văn bản liên quan, gửi đến LLM để sinh câu trả lời.
#     """
#     try:
#         # Kiểm tra nếu không có đoạn văn bản nào
#         if not request.retrieved_chunks:
#             raise HTTPException(status_code=400, detail="No relevant context found.")
#
#         # Khởi tạo LLM cục bộ dựa trên mô hình được chọn
#         llm = LocalLlms(model_name=request.model_name)
#
#         # Tạo prompt dựa trên nội dung truy xuất
#         context = "\n\n".join(request.retrieved_chunks)  # Ghép các chunk lại
#         prompt = f"""
#         Bạn là một trợ lý AI thông minh. Dưới đây là ngữ cảnh có liên quan đến câu hỏi:
#
#         --- NGỮ CẢNH ---
#         {context}
#
#         --- CÂU HỎI ---
#         {request.question}
#
#         Hãy trả lời một cách chính xác, mạch lạc và đầy đủ dựa trên thông tin trong ngữ cảnh trên.
#         Nếu không có đủ thông tin để trả lời, hãy nói 'Tôi không có đủ dữ kiện để trả lời câu hỏi này.'
#         """
#
#         # Gọi LLM để sinh câu trả lời
#         response = llm.generate_content(prompt)
#
#         return {
#             "question": request.question,
#             "answer": response.strip(),
#             "model": request.model_name
#         }
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
