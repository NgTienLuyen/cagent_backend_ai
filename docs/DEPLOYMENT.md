# Hướng Dẫn Triển Khai AI Agent Backend

Tài liệu này hướng dẫn cách triển khai AI Agent Backend lên tên miền sử dụng Docker, Nginx và SSL.

## Yêu Cầu

- Máy chủ Linux với Docker và Docker Compose được cài đặt
- Nginx đã được cài đặt
- Certbot (để cài đặt SSL)
- Tên miền đã được trỏ đến IP của máy chủ

## Bước 1: Chuẩn Bị Thư Mục Dự Án

```bash
# Tạo thư mục triển khai
mkdir -p /mnt/data/aiagent-deploy
cd /mnt/data/aiagent-deploy

# Tạo các thư mục cần thiết
mkdir -p uploads
mkdir -p uploadfiles
mkdir -p temp
```

## Bước 2: Tạo hoặc Cập Nhật Docker Compose

Tạo file `docker-compose.yml`:

```bash
nano docker-compose.yml
```

Thêm nội dung sau (thay thế các giá trị phù hợp với môi trường của bạn):

```yaml
version: '3.8'

services:
  backend:
    container_name: aiagent_backend
    image: harbor.cmc-u.edu.vn/aiagentdhs/aiagent_ai_backend:v1.1
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - shared_uploads:/app/uploads
      - ./uploadfiles:/app/uploadfiles
      - ./temp:/app/temp
    environment:
      - DB_HOST=pgvector-container
      - DB_PORT=5432
      - DB_NAME=ai_agent
      - DB_USER=postgres
      - DB_PASSWORD=newpassword
      - OCR_LANGUAGE=vie
      - OCR_DPI=600
      - OCR_CONFIG=--psm 1 --oem 3 -c textord_min_linesize=3.0 -c preserve_interword_spaces=1
      - DOCKER_ENV=true
      - TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
      - POPPLER_PATH=/usr/bin
      - TESSERACT_CMD_PATH=/usr/bin/tesseract
    depends_on:
      - postgres
    networks:
      - ai-agent-network
      - aiagent_network

  postgres:
    container_name: pgvector-container
    image: harbor.cmc-u.edu.vn/aiagentdhs/pgvector:v1.0-pg16
    restart: unless-stopped
    environment:
      - POSTGRES_DB=ai_agent
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=newpassword
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - ai-agent-network
      - aiagent_network

networks:
  ai-agent-network:
    driver: bridge
  aiagent_network:
    driver: bridge

volumes:
  pgdata:
  shared_uploads:
```

## Bước 3: Đăng Nhập vào Harbor Registry và Pull Images

```bash
# Đăng nhập vào Harbor
docker login harbor.cmc-u.edu.vn -u aiagentdhs -p Cmcuni@2025

# Pull image backend
docker pull harbor.cmc-u.edu.vn/aiagentdhs/aiagent_ai_backend:v1.1

# Pull image database (nếu cần)
docker pull harbor.cmc-u.edu.vn/aiagentdhs/pgvector:v1.0-pg16
```

## Bước 4: Khởi Động Docker Compose

```bash
cd /mnt/data/aiagent-deploy
docker-compose up -d
```

## Bước 5: Cấu Hình Nginx

Tạo file cấu hình Nginx cho tên miền:

```bash
nano /etc/nginx/conf.d/yourdomain.conf
```

Thay thế `yourdomain` bằng tên miền thực tế của bạn (ví dụ: `cagent-aibackend`).

Thêm nội dung sau:

```nginx
server {
    listen 80;
    server_name yourdomain.cmcu.edu.vn;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }

    location /uploads/ {
        alias /mnt/data/aiagent-deploy/uploads/;
        expires 1d;
        add_header Cache-Control "public";
    }
}
```

Thay thế `yourdomain.cmcu.edu.vn` bằng tên miền thực tế của bạn.

## Bước 6: Kiểm Tra và Áp Dụng Cấu Hình Nginx

```bash
# Kiểm tra cấu hình
nginx -t

# Nếu OK, reload Nginx
systemctl reload nginx
```

## Bước 7: Cài Đặt SSL với Certbot

```bash
# Cài đặt SSL
certbot --nginx -d yourdomain.cmcu.edu.vn
```

Thay thế `yourdomain.cmcu.edu.vn` bằng tên miền thực tế của bạn.

Làm theo các hướng dẫn trên màn hình để hoàn thành việc cài đặt SSL.

## Bước 8: Kiểm Tra Triển Khai

```bash
# Kiểm tra trạng thái container
docker ps

# Kiểm tra logs
docker logs aiagent_backend

# Kiểm tra kết nối HTTP
curl -I http://yourdomain.cmcu.edu.vn

# Kiểm tra kết nối HTTPS
curl -I https://yourdomain.cmcu.edu.vn

# Kiểm tra API docs
curl https://yourdomain.cmcu.edu.vn/docs
```

## Bước 9: Cấu Hình Quyền Truy Cập cho Thư Mục Uploads

```bash
# Đảm bảo quyền truy cập đúng
chown -R www-data:www-data /mnt/data/aiagent-deploy/uploads
chmod 755 /mnt/data/aiagent-deploy/uploads
```

## Khắc Phục Sự Cố

### Lỗi "host not found in upstream"

Nếu bạn gặp lỗi "host not found in upstream" trong Nginx, hãy sửa cấu hình để sử dụng `localhost` thay vì tên container:

```nginx
proxy_pass http://localhost:8000;
```

thay vì

```nginx
proxy_pass http://aiagent_backend:8000;
```

### Kiểm Tra Kết Nối Database

```bash
docker exec -it pgvector-container psql -U postgres -d ai_agent -c "SELECT 1"
```

### Xem Logs Container

```bash
docker logs aiagent_backend
```

## Cập Nhật Backend

Khi cần cập nhật backend với phiên bản mới:

1. Cập nhật phiên bản trong docker-compose.yml

```yaml
image: harbor.cmc-u.edu.vn/aiagentdhs/aiagent_ai_backend:v1.2
```

2. Pull image mới

```bash
docker pull harbor.cmc-u.edu.vn/aiagentdhs/aiagent_ai_backend:v1.2
```

3. Khởi động lại service

```bash
docker-compose down backend
docker-compose up -d backend
```

## Sao Lưu Database

```bash
docker exec -t pgvector-container pg_dumpall -c -U postgres > dump_$(date +%Y-%m-%d_%H_%M_%S).sql
```

## Kết Luận

Sau khi hoàn thành tất cả các bước trên, AI Agent Backend của bạn sẽ được triển khai thành công và có thể truy cập qua tên miền đã cấu hình với kết nối HTTPS an toàn. 