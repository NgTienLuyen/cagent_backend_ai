"""
API để xem thống kê cache và performance
"""
from fastapi import APIRouter, HTTPException
import logging
from services.query_cache import query_cache, clear_all_caches

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/cache/stats", summary="Lấy thống kê cache")
async def get_cache_stats():
    """Xem thống kê hiệu suất cache"""
    try:
        query_stats = query_cache.get_stats()
        
        # Thêm rerank cache stats
        try:
            from services.rerank_cache import rerank_cache
            rerank_stats = rerank_cache.get_stats()
        except ImportError:
            rerank_stats = {"error": "Rerank cache not available"}
        
        return {
            "success": True,
            "data": {
                "query_cache": query_stats,
                "rerank_cache": rerank_stats
            },
            "message": "Cache statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return {
            "success": False,
            "data": None,
            "message": f"Error getting cache stats: {str(e)}"
        }

@router.post("/cache/clear", summary="Xóa toàn bộ cache")
async def clear_cache():
    """Xóa toàn bộ cache để reset hiệu suất"""
    try:
        clear_all_caches()
        
        return {
            "success": True,
            "data": None,
            "message": "All caches cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return {
            "success": False,
            "data": None,
            "message": f"Error clearing cache: {str(e)}"
        }

@router.get("/rerank-cache/stats", summary="Lấy thống kê rerank cache")
async def get_rerank_cache_stats():
    """Lấy thống kê hiệu suất rerank cache"""
    try:
        from services.rerank_cache import rerank_cache
        stats = rerank_cache.get_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Rerank cache statistics retrieved successfully"
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Rerank cache module not available")
    except Exception as e:
        logger.error(f"Error getting rerank cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rerank-cache/clear", summary="Xóa rerank cache")
async def clear_rerank_cache():
    """Xóa toàn bộ rerank cache"""
    try:
        from services.rerank_cache import rerank_cache
        rerank_cache.clear()
        
        return {
            "success": True,
            "data": None,
            "message": "Rerank cache cleared successfully"
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Rerank cache module not available")
    except Exception as e:
        logger.error(f"Error clearing rerank cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/summary", summary="Tổng quan hiệu suất cache")
async def get_performance_summary():
    """Lấy tổng quan hiệu suất tất cả cache systems"""
    try:
        query_stats = query_cache.get_stats()
        
        try:
            from services.rerank_cache import rerank_cache
            rerank_stats = rerank_cache.get_stats()
        except ImportError:
            rerank_stats = {"error": "Not available"}
        
        # Tính total performance metrics
        total_requests = query_stats.get("hit_count", 0) + query_stats.get("miss_count", 0)
        rerank_requests = rerank_stats.get("hit_count", 0) + rerank_stats.get("miss_count", 0) if isinstance(rerank_stats, dict) else 0
        
        summary = {
            "total_query_requests": total_requests,
            "total_rerank_requests": rerank_requests,
            "query_cache_hit_rate": query_stats.get("hit_rate", 0),
            "rerank_cache_hit_rate": rerank_stats.get("hit_rate", 0) if isinstance(rerank_stats, dict) else 0,
            "total_cache_entries": query_stats.get("cache_size", 0) + (rerank_stats.get("cache_size", 0) if isinstance(rerank_stats, dict) else 0),
            "performance_improvement": {
                "query_cache_saves": f"{query_stats.get('hit_count', 0)} requests",
                "rerank_cache_saves": f"{rerank_stats.get('hit_count', 0) if isinstance(rerank_stats, dict) else 0} rerank operations"
            }
        }
        
        return {
            "success": True,
            "data": {
                "summary": summary,
                "details": {
                    "query_cache": query_stats,
                    "rerank_cache": rerank_stats
                }
            },
            "message": "Performance summary retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
