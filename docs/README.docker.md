# Hướng dẫn triển khai AI Agent Chatbot trên Docker

## Yêu cầu hệ thống

- Docker đã được cài đặt
- Container pgvector-container đã chạy (PostgreSQL với pgvector extension)
- Container pgadmin-container đang chạy (tùy chọn, để quản lý database)

## Các thành phần

- **aiagent_backend**: Container chứa ứng dụng FastAPI
- **pgvector-container**: Container PostgreSQL với pgvector extension
- **pgadmin-container**: Container PgAdmin 4 để quản lý database

## Quy trình triển khai

### 1. Build Docker image

```bash
docker build -t aiagent_backend .
```

Quá trình này có thể mất 5-10 phút tùy thuộc vào tốc độ mạng và cấu hình máy tính.

### 2. Khởi động container

#### Sử dụng script (khuyến nghị)

**Trên Linux/macOS:**
```bash
chmod +x run_docker.sh
./run_docker.sh
```

**Trên Windows:**
```powershell
.\run_docker.ps1
```

#### Hoặc chạy lệnh thủ công:
```bash
docker run -d \
  --name aiagent_backend \
  --restart unless-stopped \
  --network bridge \
  -p 8000:8000 \
  -v "$(pwd)/uploadfiles:/app/uploadfiles" \
  -v "$(pwd)/temp:/app/temp" \
  -e DB_HOST=pgvector-container \
  -e DB_PORT=5432 \
  -e DB_NAME=ai_agent \
  -e DB_USER=postgres \
  -e DB_PASSWORD=newpassword \
  -e OCR_LANGUAGE=vie \
  -e OCR_DPI=600 \
  -e 'OCR_CONFIG=--psm 1 --oem 3 -c textord_min_linesize=3.0 -c preserve_interword_spaces=1' \
  aiagent_backend
```

### 3. Kiểm tra logs

```bash
docker logs -f aiagent_backend
```

### 4. Truy cập API

API sẽ khả dụng tại: http://localhost:8000

- API Docs: http://localhost:8000/docs
- Chunking API: http://localhost:8000/api/upload/chunking/
- Direct Chat API: http://localhost:8000/api/direct-llm/

## Quản lý Container

### Dừng container
```bash
docker stop aiagent_backend
```

### Khởi động lại container
```bash
docker start aiagent_backend
```

### Xóa container
```bash
docker rm -f aiagent_backend
```

## Xử lý sự cố

### 1. Lỗi kết nối database

Nếu gặp lỗi kết nối đến database, hãy đảm bảo:

- Container pgvector-container đang chạy: `docker ps | grep pgvector-container`
- Thông tin kết nối (host, port, username, password) đúng
- Kiểm tra xem containers có chung mạng không: `docker network inspect bridge`

### 2. Lỗi OCR

Nếu OCR không hoạt động:

- Kiểm tra Tesseract OCR đã được cài đặt đúng trong container
- Kiểm tra giá trị biến môi trường OCR_LANGUAGE, OCR_DPI, OCR_CONFIG
- Vào container và kiểm tra thủ công: `docker exec -it aiagent_backend bash`

### 3. Các lỗi khác

- Kiểm tra logs: `docker logs aiagent_backend`
- Truy cập vào container: `docker exec -it aiagent_backend bash`
- Kiểm tra cấu hình: `docker exec aiagent_backend env | grep DB_`

## Cấu hình nâng cao

### Thay đổi cổng
```bash
docker run -d --name aiagent_backend -p 9000:8000 ... aiagent_backend
```

### Sử dụng file .env
```bash
docker run -d --name aiagent_backend --env-file .env.docker ... aiagent_backend
```

### Nếu cần truy cập qua HTTPS
Thêm Nginx hoặc Traefik làm reverse proxy để xử lý HTTPS. 