#!/usr/bin/env python3
"""
Model Preload Checker - Kiểm tra models đã được preload chưa
"""

import requests
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api_health():
    """Kiểm tra API health"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)

def test_embedding_endpoint():
    """Test embedding endpoint để trigger model loading"""
    try:
        test_data = {
            "text": "test embedding model loading",
            "knowledgeBaseId": "test-kb-id"
        }
        
        response = requests.post(
            'http://localhost:8000/api/embedding/generate',
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Embedding test thành công: {len(result.get('embedding', []))} dimensions")
            return True
        else:
            logger.error(f"❌ Embedding test thất bại: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi khi test embedding: {e}")
        return False

def test_reranker_endpoint():
    """Test reranker endpoint để trigger model loading"""
    try:
        test_data = {
            "query": "test query",
            "chunks": [
                {"chunk_text": "test chunk 1", "chunk_id": "1"},
                {"chunk_text": "test chunk 2", "chunk_id": "2"}
            ],
            "knowledgeBaseId": "test-kb-id"
        }
        
        response = requests.post(
            'http://localhost:8000/api/search/rerank',
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Reranker test thành công: {len(result.get('reranked_chunks', []))} chunks")
            return True
        else:
            logger.error(f"❌ Reranker test thất bại: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi khi test reranker: {e}")
        return False

def monitor_model_loading():
    """Monitor quá trình load models"""
    logger.info("🔍 Bắt đầu kiểm tra model loading...")
    
    # Kiểm tra API health
    api_healthy, api_response = check_api_health()
    if not api_healthy:
        logger.error(f"❌ API không sẵn sàng: {api_response}")
        return False
    
    logger.info(f"✅ API sẵn sàng: {api_response}")
    
    # Test embedding endpoint
    logger.info("🔄 Testing embedding endpoint...")
    embedding_success = test_embedding_endpoint()
    
    # Test reranker endpoint  
    logger.info("🔄 Testing reranker endpoint...")
    reranker_success = test_reranker_endpoint()
    
    # Tổng kết
    if embedding_success and reranker_success:
        logger.info("🎉 Tất cả models đã được load thành công!")
        return True
    else:
        logger.warning("⚠️ Một số models chưa được load hoàn toàn")
        return False

def main():
    """Main function"""
    logger.info("🚀 Model Preload Checker")
    logger.info("=" * 50)
    
    # Chờ API khởi động
    logger.info("⏳ Chờ API khởi động...")
    time.sleep(10)
    
    # Monitor model loading
    success = monitor_model_loading()
    
    if success:
        logger.info("✅ Kiểm tra hoàn tất - Models đã sẵn sàng!")
    else:
        logger.warning("⚠️ Kiểm tra hoàn tất - Có thể cần thời gian để load models")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
