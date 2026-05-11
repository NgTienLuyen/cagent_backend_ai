# AI Agent Backend - Memory Optimization

## 🎯 **Vấn đề đã giải quyết:**
Hệ thống bị restart liên tục do memory overflow từ việc load quá nhiều models cùng lúc.

## 🔧 **Giải pháp đã triển khai:**

### **1. Tối ưu Model Loading:**
- ✅ **Bỏ model**: `cross-encoder/ms-marco-MiniLM-L-2-v2` 
- ✅ **Giữ lại**: `BAAI/bge-reranker-base` (model chính)
- ✅ **Giảm memory**: Từ ~2GB xuống ~1GB cho models

### **2. Cấu hình linh hoạt theo môi trường:**

#### **Local Development:**
```bash
.\deploy_flexible.ps1 -Environment local
```
- Workers: 2
- Memory limit: 3GB  
- Models: 1 reranker model
- **Tổng memory**: ~2GB

#### **Production:**
```bash
.\deploy_flexible.ps1 -Environment production
```
- Workers: 4
- Memory limit: 8GB
- Models: 1 reranker model  
- **Tổng memory**: ~4GB

### **3. Environment Variables:**
```bash
# Dockerfile sử dụng environment variables
WORKERS=${WORKERS:-4}           # Mặc định 4 workers
MAX_REQUESTS=${MAX_REQUESTS:-1000}
TIMEOUT=${TIMEOUT:-300}
ENVIRONMENT=${ENVIRONMENT:-local}
SKIP_MODEL_PRELOAD=${SKIP_MODEL_PRELOAD:-false}
```

## 📊 **So sánh Memory Usage:**

| Cấu hình | Workers | Models | Memory/Worker | Tổng Memory |
|----------|---------|--------|---------------|--------------|
| **Trước** | 4 | 2 models | ~2GB | ~8GB |
| **Sau Local** | 2 | 1 model | ~1GB | ~2GB |
| **Sau Production** | 4 | 1 model | ~1GB | ~4GB |

## 🚀 **Cách sử dụng:**

### **1. Deploy cho Local Development:**
```bash
# Windows PowerShell
.\deploy_flexible.ps1 -Environment local

# Linux/Mac
./deploy_flexible.sh local
```

### **2. Deploy cho Production:**
```bash
# Windows PowerShell  
.\deploy_flexible.ps1 -Environment production

# Linux/Mac
./deploy_flexible.sh production
```

### **3. Monitor Memory:**
```bash
python monitor_memory.py
```

### **4. Xem Logs:**
```bash
# Local
docker-compose -f docker-compose.local.yml logs -f

# Production  
docker-compose -f docker-compose.production.yml logs -f
```

## ⚠️ **Lưu ý quan trọng:**

1. **Model Performance**: Chỉ sử dụng 1 reranker model có thể giảm độ chính xác reranking một chút, nhưng vẫn đủ tốt cho hầu hết use cases.

2. **Memory Monitoring**: Sử dụng `monitor_memory.py` để theo dõi memory usage realtime.

3. **Health Checks**: Containers có health checks tự động restart nếu có vấn đề.

4. **Graceful Degradation**: Nếu memory thấp, hệ thống sẽ skip warmup models.

## 🔍 **Troubleshooting:**

### **Nếu vẫn bị restart:**
1. Kiểm tra memory usage: `python monitor_memory.py`
2. Giảm workers xuống 1: `WORKERS=1`
3. Skip model preload: `SKIP_MODEL_PRELOAD=true`

### **Nếu performance thấp:**
1. Tăng memory limits trong docker-compose
2. Sử dụng production config: `.\deploy_flexible.ps1 -Environment production`
3. Monitor với `docker stats`

## 📈 **Kết quả mong đợi:**
- ✅ Giảm restart frequency từ liên tục xuống 0
- ✅ Giảm memory usage từ 8GB xuống 2-4GB  
- ✅ Tăng stability và uptime
- ✅ Dễ dàng chuyển đổi giữa local và production
