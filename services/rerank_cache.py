"""
Rerank Cache cho BAAI/bge-reranker-base - Cache kết quả reranking để tránh tính toán lại
"""
import hashlib
import json
import time
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class RerankCache:
    """In-memory cache cho reranking results"""
    
    def __init__(self, max_size: int = 2000, ttl_seconds: int = 10800):  # 3 giờ TTL, cache lớn hơn
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
    
    def _generate_cache_key(self, query: str, chunk_ids: List[str]) -> str:
        """Tạo cache key từ query và chunk IDs"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Sort chunk IDs để đảm bảo consistent key
        sorted_chunk_ids = sorted(chunk_ids)
        
        cache_data = {
            "query": normalized_query,
            "chunk_ids": sorted_chunk_ids,
            "model": "BAAI/bge-reranker-base"  # Specific cho model này
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    
    def get(self, query: str, chunk_ids: List[str]) -> Optional[Dict[str, float]]:
        """Lấy rerank scores từ cache"""
        cache_key = self._generate_cache_key(query, chunk_ids)
        
        if cache_key not in self.cache:
            self.miss_count += 1
            return None
        
        entry = self.cache[cache_key]
        
        # Kiểm tra TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self.cache[cache_key]
            self.miss_count += 1
            logger.debug(f"Rerank cache entry expired for key: {cache_key[:8]}...")
            return None
        
        # Cache hit
        self.hit_count += 1
        entry["last_accessed"] = time.time()
        
        logger.info(f"🎯 Rerank Cache HIT for query: '{query[:30]}...' ({len(chunk_ids)} chunks)")
        return entry["scores"]
    
    def put(self, query: str, chunk_ids: List[str], scores: Dict[str, float]) -> None:
        """Lưu rerank scores vào cache"""
        cache_key = self._generate_cache_key(query, chunk_ids)
        
        # Nếu cache đầy, xóa entry cũ nhất
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = {
            "scores": scores,
            "timestamp": time.time(),
            "last_accessed": time.time()
        }
        
        logger.info(f"💾 Cached rerank scores for query: '{query[:30]}...' ({len(chunk_ids)} chunks)")
    
    def _evict_oldest(self):
        """Xóa cache entry cũ nhất"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k]["last_accessed"])
        del self.cache[oldest_key]
        logger.debug(f"Evicted oldest rerank cache entry: {oldest_key[:8]}...")
    
    def clear(self):
        """Xóa toàn bộ cache"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("🗑️  Rerank cache cleared")
    
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
            "ttl_seconds": self.ttl_seconds,
            "model": "BAAI/bge-reranker-base"
        }

# Global rerank cache instance
rerank_cache = RerankCache(max_size=2000, ttl_seconds=10800)  # 3 giờ TTL, cache lớn hơn
