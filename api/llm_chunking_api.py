# from fastapi import APIRouter, HTTPException, UploadFile, File
# from pydantic import BaseModel
# from typing import List
# from llms.localLllms import run_ollama_model
# import aiofiles
# import re
#
# # Khởi tạo router
# router = APIRouter()
#
# # Khởi tạo model LLM cục bộ
# local_llm = run_ollama_model(model_name="DeepSeek R1 1.5B")  # Có thể thay bằng model khác
#
# async def read_file(file: UploadFile) -> str:
#     """Đọc nội dung từ file tải lên."""
#     contents = ""
#     async with aiofiles.open(file.filename, 'r', encoding='utf-8') as f:
#         contents = await f.read()
#     return contents
#
# @router.post("/chunk_text/")
# async def chunk_text(file: UploadFile = File(...)):
#     """
#     API nhận file văn bản và chia thành các chunk có ý nghĩa bằng cách sử dụng Local LLM.
#     """
#     try:
#         text = await read_file(file)
#         if not text.strip():
#             raise HTTPException(status_code=400, detail="Input text cannot be empty.")
#
#         # Tạo prompt yêu cầu LLM xác định vị trí chia chunk
#         system_prompt = (
#             "You are an AI assistant trained to split text into semantically meaningful sections. "
#             "Each section should contain information about a specific topic. "
#             "Your task is to analyze the provided text and identify where topic shifts occur. "
#             "Return 'split_after: X, Y' where X and Y are the chunk numbers after which a split should occur. "
#             "Ensure that the response includes at least two split points."
#         )
#
#         messages = [
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": f"CHUNKED_TEXT: {text}\n\nIdentify topic transitions and return the chunk numbers."
#             }
#         ]
#
#         # Gửi yêu cầu tới Local LLM
#         result_string = local_llm.create_agentic_chunker_message(
#             system_prompt=system_prompt,
#             messages=messages,
#             max_tokens=200,
#             temperature=0.2
#         )
#
#         # Debug output
#         print(f"🔥 Local LLM Response: {result_string}")
#
#         # Trích xuất số thứ tự nơi cần chia chunk
#         split_after_line = [line for line in result_string.split("\n") if "split_after:" in line]
#         if not split_after_line:
#             print("⚠️ No valid split points detected. Using default behavior.")
#             return {"total_chunks": 1, "chunks": [text]}
#
#         split_indices = list(map(int, re.findall(r"\d+", split_after_line[0])))
#
#         # Tách văn bản thành các đoạn theo chỉ dẫn của LLM
#         sentences = text.split(". ")
#         chunks = []
#         current_chunk = ""
#
#         for i, sentence in enumerate(sentences):
#             current_chunk += sentence + ". "
#             if i in split_indices:
#                 chunks.append(current_chunk.strip())
#                 current_chunk = ""
#
#         if current_chunk:
#             chunks.append(current_chunk.strip())
#
#         return {
#             "total_chunks": len(chunks),
#             "chunks": chunks
#         }
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
