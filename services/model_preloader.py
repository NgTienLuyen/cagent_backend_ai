"""
Model Preloader - Tải trước tất cả models khi khởi động app
Tối ưu hóa để tránh memory leak và restart liên tục
"""
import asyncio
import logging
import os
from typing import Optional
try:
    import psutil
except Exception:  # psutil có thể chưa cài khi chạy ngoài Docker
    psutil = None
from llms.config_loader import ModelConfigLoader
from services.query_service import GlobalStateManager

logger = logging.getLogger(__name__)

class ModelPreloader:
    """Preload tất cả models cần thiết khi khởi động"""
    
    # Flag để kiểm tra xem đã preload chưa
    _preload_completed = False
    _preload_lock = asyncio.Lock()
    
    @staticmethod
    async def preload_all_models():
        """Preload tất cả models quan trọng với kiểm tra duplicate"""
        async with ModelPreloader._preload_lock:
            if ModelPreloader._preload_completed:
                logger.info("🚀 Models đã được preload trước đó, bỏ qua")
                return
            
            # Kiểm tra environment variable để skip preload
            skip_preload = os.getenv('SKIP_MODEL_PRELOAD', 'false').lower() == 'true'
            if skip_preload:
                logger.info("⏭️ Skip model preload theo environment variable SKIP_MODEL_PRELOAD=true")
                ModelPreloader._preload_completed = True
                return
            
            # Kiểm tra cơ chế Model Server
            use_model_server = os.getenv('USE_MODEL_SERVER', 'false').lower() == 'true'
            if use_model_server:
                logger.info("🔄 Sử dụng Model Server - chỉ Worker 1 load models")
                from services.model_server import initialize_model_server
                initialize_model_server()
                ModelPreloader._preload_completed = True
                return
                
            logger.info("🚀 Bắt đầu preload models...")
            
            try:
                # Preload TẤT CẢ models cần thiết để tránh lazy loading
                global_state = GlobalStateManager()
                
                # 1. Preload reranker models
                reranker_models = [
                    "BAAI/bge-reranker-base"  # Chỉ giữ lại model này
                ]
                
                logger.info("🔄 Bắt đầu preload reranker models...")
                for model_name in reranker_models:
                    try:
                        logger.info(f"🔄 Đang tải reranker model: {model_name}")
                        await global_state.get_reranker(model_name)
                        logger.info(f"✅ Đã tải xong reranker model: {model_name}")
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"❌ Lỗi khi tải reranker model {model_name}: {e}")
                        continue
                
                # 2. Preload embedding models
                embedding_models = [
                    {
                        "model_name": "BAAI/bge-base-en-v1.5",
                        "model_type": "local",
                        "provider": "sentencetransformer"
                    }
                ]
                
                logger.info("🔄 Bắt đầu preload embedding models...")
                for embedding_config in embedding_models:
                    try:
                        logger.info(f"🔄 Đang tải embedding model: {embedding_config['model_name']}")
                        
                        # Tạo config cho embedding model
                        llm_config = {
                            "model_type": embedding_config["model_type"],
                            "embedding_model": embedding_config["model_name"],
                            "embedding_provider": embedding_config["provider"]
                        }
                        
                        # Preload embedding model
                        from llms.config_loader import ModelConfigLoader
                        embedding_model = ModelConfigLoader.load_embedding_model(llm_config)
                        
                        # Test model với dummy text
                        dummy_text = "test embedding"
                        test_embedding = embedding_model.generate_embedding(dummy_text)
                        
                        logger.info(f"✅ Đã tải xong embedding model: {embedding_config['model_name']} (dim: {len(test_embedding)})")
                        await asyncio.sleep(2)  # Delay dài hơn cho embedding models
                        
                    except Exception as e:
                        logger.error(f"❌ Lỗi khi tải embedding model {embedding_config['model_name']}: {e}")
                        continue
                
                ModelPreloader._preload_completed = True
                logger.info("🎉 Preload models hoàn tất!")
                
            except Exception as e:
                logger.error(f"❌ Lỗi trong quá trình preload models: {e}")
                # Không đánh dấu completed nếu có lỗi để có thể retry

    @staticmethod 
    async def warmup_models():
        """Warmup models với dummy queries - chỉ khi cần thiết"""
        try:
            # Chỉ warmup nếu có memory đủ
            available_gb: Optional[float] = None
            # Ưu tiên psutil (cross-platform)
            if psutil is not None:
                available_gb = psutil.virtual_memory().available / (1024**3)
            else:
                # Fallback an toàn nếu không có psutil (Windows không hỗ trợ os.sysconf)
                available_gb = None

            if available_gb is not None and available_gb < 4:  # Nếu RAM < 4GB thì skip warmup
                logger.info("💾 RAM thấp, bỏ qua warmup models")
                return
                
            logger.info("🔥 Warming up models...")
            
            global_state = GlobalStateManager()
            
            # Warmup reranker với dummy query đơn giản
            try:
                tokenizer, model = await global_state.get_reranker("BAAI/bge-reranker-base")
                if tokenizer and model:
                    # Dummy rerank test với batch nhỏ
                    dummy_pairs = [["test", "test"]]
                    if hasattr(model, 'predict'):
                        model.predict(dummy_pairs, show_progress_bar=False)
                    logger.info("✅ Reranker warmed up")
            except Exception as e:
                logger.warning(f"⚠️ Warmup reranker failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Model warmup error: {e}")
