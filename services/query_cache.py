"""
Query Cache Service - Cache kết quả query để tối ưu hiệu suất
"""
import hashlib
import json
import time
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

class QueryCache:
    """In-memory cache cho query results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Args:
            max_size: Số lượng cache entries tối đa
            ttl_seconds: Thời gian sống của cache entry (giây)
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_cache_key(self, query: str, knowledge_base_id: str, config_id: str) -> str:
        """Tạo cache key từ query và các tham số"""
        # Normalize query - lowercase và strip whitespace
        normalized_query = query.lower().strip()
        
        # Tạo hash từ các tham số quan trọng
        cache_data = {
            "query": normalized_query,
            "knowledge_base_id": knowledge_base_id,
            "config_id": config_id
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    
    def get(self, query: str, knowledge_base_id: str, config_id: str) -> Optional[Dict[str, Any]]:
        """Lấy kết quả từ cache"""
        cache_key = self._generate_cache_key(query, knowledge_base_id, config_id)
        
        if cache_key not in self.cache:
            self.miss_count += 1
            return None
        
        entry = self.cache[cache_key]
        
        # Kiểm tra TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self.cache[cache_key]
            self.miss_count += 1
            logger.debug(f"Cache entry expired for key: {cache_key[:8]}...")
            return None
        
        # Cache hit
        self.hit_count += 1
        entry["last_accessed"] = time.time()
        
        logger.info(f"🎯 Cache HIT for query: '{query[:50]}...' (key: {cache_key[:8]})")
        return entry["data"]
    
    def put(self, query: str, knowledge_base_id: str, config_id: str, result: Dict[str, Any]) -> None:
        """Lưu kết quả vào cache"""
        cache_key = self._generate_cache_key(query, knowledge_base_id, config_id)
        
        # Nếu cache đầy, xóa entry cũ nhất
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = {
            "data": result,
            "timestamp": time.time(),
            "last_accessed": time.time()
        }
        
        logger.info(f"💾 Cached result for query: '{query[:50]}...' (key: {cache_key[:8]})")
    
    def _evict_oldest(self):
        """Xóa cache entry cũ nhất"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k]["last_accessed"])
        del self.cache[oldest_key]
        logger.debug(f"Evicted oldest cache entry: {oldest_key[:8]}...")
    
    def clear(self):
        """Xóa toàn bộ cache"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("🗑️  Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 2),
            "ttl_seconds": self.ttl_seconds
        }

# Global cache instance
query_cache = QueryCache(max_size=500, ttl_seconds=1800)  # 30 phút TTL

@lru_cache(maxsize=200)
def get_embedding_cache(query_hash: str):
    """LRU cache cho embeddings - decorator cache"""
    # Dummy function, actual caching sẽ được implement trong embedding service
    pass

def clear_all_caches():
    """Clear tất cả caches"""
    query_cache.clear()
    get_embedding_cache.cache_clear()
    
    # Clear rerank cache nếu có
    try:
        from services.rerank_cache import rerank_cache
        rerank_cache.clear()
        logger.info("🗑️  All caches cleared (including rerank cache)")
    except ImportError:
        logger.info("🗑️  All caches cleared (rerank cache not available)")
    except Exception as e:
        logger.warning(f"Error clearing rerank cache: {e}")
        logger.info("🗑️  Query and embedding caches cleared")
