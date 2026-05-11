from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import logging
import os
import sys

# Cấu hình encoding cho stdout để hỗ trợ tiếng Việt
sys.stdout.reconfigure(encoding='utf-8')

# Thiết lập logging cho ứng dụng
logging_level = os.getenv("LOGGING_LEVEL", "INFO")
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, logging_level),
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Log ra console
        logging.FileHandler("app.log", encoding='utf-8')  # Log ra file với utf-8
    ]
)

# Thiết lập log cho các module cụ thể 
logging.getLogger("chunking").setLevel(logging.INFO)
logging.getLogger("api").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.info("Khởi động ứng dụng...")

# Khởi tạo FastAPI
app = FastAPI(
    title="CMC Chatbot AI Backend",
    description="Backend API cho hệ thống chatbot AI của CMC",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Chunking", "description": "API xử lý chunking văn bản"},
        {"name": "Embedding", "description": "API xử lý embedding và vector"},
        {"name": "Search", "description": "API tìm kiếm semantic"},
        {"name": "CRUD", "description": "API thao tác CRUD với chunks"},
        {"name": "Direct_chat", "description": "API chat trực tiếp với LLM"},
        {"name": "Summarization", "description": "API tóm tắt văn bản"}
    ]
)

# Cấu hình CORS với đầy đủ chi tiết
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn gốc (origins)
    allow_credentials=True,  # Cho phép gửi cookies trong CORS requests
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các header
    expose_headers=["X-Request-ID", "X-Process-Time", "Content-Type"],  # Thêm các header hiển thị cho client
    max_age=600,  # Thời gian cache cho preflight requests (seconds)
)

# Preload models khi startup
@app.on_event("startup")
async def startup_event():
    """Preload models để tối ưu hiệu suất với kiểm tra memory"""
    try:
        # Reset cooldown khi restart server
        from llms.config_loader import reset_cooldowns
        reset_cooldowns()
        
        # Kiểm tra memory trước khi preload
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"💾 Available memory: {available_memory_gb:.2f} GB")
        
        if available_memory_gb < 2:
            logger.warning("⚠️ Memory thấp (< 2GB), bỏ qua preload models")
            return
        
        logger.info("🚀 Bắt đầu preload models...")
        from services.model_preloader import ModelPreloader
        
        # Preload models với kiểm tra duplicate
        await ModelPreloader.preload_all_models()
        
        # Chỉ warmup nếu có đủ memory
        if available_memory_gb >= 4:
            await ModelPreloader.warmup_models()
        else:
            logger.info("💾 Memory < 4GB, bỏ qua warmup")
        
        logger.info("✅ Preload models hoàn tất!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi preload models: {e}")
        # Không fail app start nếu preload lỗi

# Bao gồm các router từ các file API riêng biệt
from api.semantic_chunking_api import router as chunking_router
from api.chunks_embedding_api import router as embedding_router
from api.chunks_search_api import router as search_router
from api.crud_chunks_api import router as crud
from api.direct_chat_api import router as direct_chat_router
from api.summarization_api import router as summarization_router
from api.cache_stats_api import router as cache_router
#from api.model_management_api import router as model_management_router 

@app.get("/", tags=["Health Check"])
async def root():
    """Endpoint kiểm tra sức khỏe của API"""
    return {"message": "CMC Chatbot AI Backend đang hoạt động!", "status": "healthy"}

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Endpoint kiểm tra sức khỏe chi tiết"""
    return {
        "status": "healthy",
        "timestamp": "2025-08-21T15:51:00",
        "version": "1.0.0",
        "services": ["chunking", "embedding", "search", "crud", "direct_chat", "summarization"]
    }

app.include_router(chunking_router, prefix="/api", tags=["Chunking"])
app.include_router(embedding_router, prefix="/api", tags=["Embedding"])
app.include_router(search_router, prefix="/api", tags=["Search"])
app.include_router(crud, prefix="/api", tags=["CRUD"])
app.include_router(direct_chat_router, prefix="/api", tags=["Direct_chat"])
app.include_router(summarization_router, prefix="/api", tags=["Summarization"])
app.include_router(cache_router, prefix="/api", tags=["Performance"])
#app.include_router(model_management_router, prefix="/api", tags=["model_management"])
