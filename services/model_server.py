"""
Model Server - Chỉ Worker 1 load models và serve cho các workers khác
Giải pháp tối ưu memory: 1 worker load, 4 workers dùng chung
"""
import asyncio
import logging
import os
import json
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn
from threading import Thread
import requests
from llms.config_loader import ModelConfigLoader
from services.query_service import GlobalStateManager

logger = logging.getLogger(__name__)

class ModelServer:
    """Model Server chạy trong Worker 1 để serve models cho các workers khác"""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.app = FastAPI(title="Model Server")
        self.models_loaded = False
        self.global_state = None
        self._setup_routes()
    
    def _setup_routes(self):
        """Thiết lập các API routes"""
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "models_loaded": self.models_loaded}
        
        @self.app.post("/load_models")
        async def load_models():
            """Load tất cả models"""
            try:
                if self.models_loaded:
                    return {"status": "already_loaded"}
                
                logger.info("🔄 Worker 1: Bắt đầu load models...")
                self.global_state = GlobalStateManager()
                
                # Load reranker model
                await self.global_state.get_reranker("BAAI/bge-reranker-base")
                
                # Load embedding model
                llm_config = {
                    "model_type": "local",
                    "embedding_model": "BAAI/bge-base-en-v1.5",
                    "embedding_provider": "sentencetransformer"
                }
                self.embedding_model = ModelConfigLoader.load_embedding_model(llm_config)
                
                self.models_loaded = True
                logger.info("✅ Worker 1: Models đã load xong!")
                
                return {"status": "loaded", "models": ["reranker", "embedding"]}
                
            except Exception as e:
                logger.error(f"❌ Worker 1: Lỗi load models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/rerank")
        async def rerank(pairs: list):
            """Rerank các cặp văn bản"""
            if not self.models_loaded:
                raise HTTPException(status_code=503, detail="Models not loaded")
            
            try:
                tokenizer, model = await self.global_state.get_reranker("BAAI/bge-reranker-base")
                if not tokenizer or not model:
                    raise HTTPException(status_code=503, detail="Reranker not available")
                
                # Thực hiện reranking
                scores = model.predict(pairs, show_progress_bar=False)
                return {"scores": scores.tolist()}
                
            except Exception as e:
                logger.error(f"❌ Worker 1: Lỗi reranking: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/embed")
        async def embed(texts: list):
            """Tạo embeddings cho danh sách văn bản"""
            if not self.models_loaded:
                raise HTTPException(status_code=503, detail="Models not loaded")
            
            try:
                embeddings = []
                for text in texts:
                    embedding = self.embedding_model.generate_embedding(text)
                    embeddings.append(embedding.tolist())
                
                return {"embeddings": embeddings}
                
            except Exception as e:
                logger.error(f"❌ Worker 1: Lỗi embedding: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def start(self):
        """Khởi động model server"""
        logger.info(f"🚀 Worker 1: Khởi động Model Server trên port {self.port}")
        
        # Chạy server trong thread riêng
        def run_server():
            uvicorn.run(
                self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="info",
                access_log=False
            )
        
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Đợi server khởi động
        time.sleep(2)
        
        # Load models ngay lập tức
        asyncio.create_task(self._load_models_async())
    
    async def _load_models_async(self):
        """Load models bất đồng bộ"""
        try:
            await self.load_models()
        except Exception as e:
            logger.error(f"❌ Worker 1: Lỗi load models async: {e}")

class ModelClient:
    """Model Client cho các workers khác để gọi đến Model Server"""
    
    def __init__(self, model_server_url: str = "http://localhost:8001"):
        self.model_server_url = model_server_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    async def rerank(self, pairs: list) -> list:
        """Gọi reranking đến Model Server"""
        try:
            response = self.session.post(
                f"{self.model_server_url}/rerank",
                json=pairs,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["scores"]
            
        except Exception as e:
            logger.error(f"❌ Model Client: Lỗi reranking: {e}")
            # Fallback: trả về scores mặc định
            return [0.5] * len(pairs)
    
    async def embed(self, texts: list) -> list:
        """Gọi embedding đến Model Server"""
        try:
            response = self.session.post(
                f"{self.model_server_url}/embed",
                json=texts,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["embeddings"]
            
        except Exception as e:
            logger.error(f"❌ Model Client: Lỗi embedding: {e}")
            # Fallback: trả về embeddings mặc định
            return [[0.0] * 768] * len(texts)
    
    def health_check(self) -> bool:
        """Kiểm tra Model Server có hoạt động không"""
        try:
            response = self.session.get(f"{self.model_server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Global instances
model_server = None
model_client = None

def initialize_model_server():
    """Khởi tạo Model Server (chỉ Worker 1)"""
    global model_server
    
    # Chỉ Worker 1 khởi tạo Model Server
    worker_id = os.getenv('WORKER_ID', '1')
    if worker_id == '1':
        logger.info("🚀 Worker 1: Khởi tạo Model Server")
        model_server = ModelServer()
        model_server.start()
    else:
        logger.info(f"🚀 Worker {worker_id}: Không khởi tạo Model Server")

def get_model_client():
    """Lấy Model Client instance"""
    global model_client
    
    if model_client is None:
        model_client = ModelClient()
    
    return model_client
