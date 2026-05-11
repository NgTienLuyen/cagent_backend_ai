import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

# Cache để lưu trữ các model đã load
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

class LocalEmbedding:
    """Local embedding sử dụng SentenceTransformer với caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Khởi tạo local embedding model
        
        Args:
            model_name: Tên model SentenceTransformer (mặc định: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.model = self._get_or_load_model()
    
    def _get_or_load_model(self) -> SentenceTransformer:
        """Lấy model từ cache hoặc load mới nếu chưa có"""
        global _MODEL_CACHE
        
        if self.model_name in _MODEL_CACHE:
            logger.info(f"[LOCAL_EMBEDDING] Sử dụng model từ cache: {self.model_name}")
            return _MODEL_CACHE[self.model_name]
        
        try:
            logger.info(f"[LOCAL_EMBEDDING] Đang tải model mới: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            
            # Lưu vào cache
            _MODEL_CACHE[self.model_name] = model
            logger.info(f"[LOCAL_EMBEDDING] Đã tải và cache model: {self.model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"[LOCAL_EMBEDDING] Lỗi khi tải model {self.model_name}: {str(e)}")
            raise
    
    @classmethod
    def clear_cache(cls):
        """Xóa tất cả model khỏi cache"""
        global _MODEL_CACHE
        _MODEL_CACHE.clear()
        logger.info("[LOCAL_EMBEDDING] Đã xóa tất cả model khỏi cache")
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, str]:
        """Lấy thông tin về các model đã cache"""
        global _MODEL_CACHE
        return {name: "cached" for name in _MODEL_CACHE.keys()}
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Tạo embedding cho text
        
        Args:
            text: Văn bản cần tạo embedding
            
        Returns:
            List[float]: Vector embedding đã được normalize
        """
        try:
            # Model đã được load trong __init__
            # normalize_embeddings=True để đảm bảo cosine similarity chính xác
            embedding = self.model.encode(text, convert_to_tensor=False, normalize_embeddings=False)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"[LOCAL_EMBEDDING] Lỗi khi tạo embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho nhiều text cùng lúc
        
        Args:
            texts: Danh sách văn bản cần tạo embedding
            
        Returns:
            List[List[float]]: Danh sách vector embedding đã được normalize
        """
        try:
            # Model đã được load trong __init__
            # normalize_embeddings=True để đảm bảo cosine similarity chính xác
            embeddings = self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"[LOCAL_EMBEDDING] Lỗi khi tạo embeddings: {str(e)}")
            raise

# Danh sách các model SentenceTransformer phổ biến
AVAILABLE_LOCAL_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",  # 384 dimensions, nhanh, nhẹ
    "all-mpnet-base-v2": "all-mpnet-base-v2",  # 768 dimensions, chất lượng cao
    "paraphrase-multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2",  # Hỗ trợ đa ngôn ngữ
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2",
}
