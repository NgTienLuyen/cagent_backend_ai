import asyncio
import json
from langchain.callbacks.base import BaseCallbackHandler
from fastapi.responses import StreamingResponse
import logging

logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    """Handler gọi callback từ LangChain để streaming token"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.stop_signal = False
        self.metadata = {}
        
    async def on_llm_new_token(self, token: str, **kwargs):
        """Được gọi khi có token mới từ LLM"""
        await self.queue.put(token)
    
    async def on_llm_start(self, serialized, prompts, **kwargs):
        """Được gọi khi LLM bắt đầu xử lý"""
        logger.info(f"LLM bắt đầu xử lý prompt...")
    
    async def on_llm_end(self, response, **kwargs):
        """Được gọi khi LLM kết thúc xử lý"""
        logger.info(f"LLM kết thúc xử lý")
        await self.queue.put(None)  # signal the end of generation
    
    async def on_llm_error(self, error, **kwargs):
        """Được gọi khi LLM gặp lỗi"""
        logger.error(f"LLM gặp lỗi: {str(error)}")
        await self.queue.put(None)  # signal the end of generation
    
    def set_metadata(self, metadata):
        """Lưu metadata để gửi cuối stream"""
        self.metadata = metadata
        
async def generate_tokens(handler):
    """Generator để tạo streaming response"""
    try:
        # Gửi token từ queue khi có
        while True:
            token = await handler.queue.get()
            if token is None:  # Tín hiệu kết thúc stream
                # Gửi kết quả metadata nếu có
                if handler.metadata:
                    yield f"data: {json.dumps({'token': '', 'finished': True, 'metadata': handler.metadata})}\n\n"
                else:
                    yield f"data: {json.dumps({'token': '', 'finished': True})}\n\n"
                break
                
            # Gửi token bình thường
            yield f"data: {json.dumps({'token': token, 'finished': False})}\n\n"
    except Exception as e:
        logger.error(f"Lỗi khi generate token: {str(e)}")
        yield f"data: {json.dumps({'token': '', 'error': str(e), 'finished': True})}\n\n"

def create_streaming_response(handler):
    """Tạo FastAPI StreamingResponse từ handler"""
    return StreamingResponse(
        generate_tokens(handler),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"  # Ngăn nginx buffer
        }
    )