# Hướng Dẫn Cài Đặt PostgreSQL và pgvector trên Docker

Tài liệu này hướng dẫn chi tiết cách tải và cài đặt PostgreSQL với extension pgvector trên Docker cho dự án AI Agent Backend.

## Mục Lục
1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Cài Đặt Docker](#cài-đặt-docker)
3. [Tạo Docker Image PostgreSQL vs pgvector](#tạo-docker-image-postgresql-với-pgvector)
5. [Khởi Động và Kiểm Tra](#khởi-động-và-kiểm-tra)
6. [Kết Nối và Sử Dụng](#kết-nối-và-sử-dụng)
7. [Khắc Phục Sự Cố](#khắc-phục-sự-cố)
8. [Sao Lưu và Phục Hồi](#sao-lưu-và-phục-hồi)

## Yêu Cầu Hệ Thống

- **Docker Engine** phiên bản 20.10 trở lên
- **Docker Compose** phiên bản 2.0 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk**: Tối thiểu 10GB trống
- **OS**: Linux, macOS, hoặc Windows với WSL2

## I. Cài Đặt Docker

### Trên Ubuntu/Debian
```bash
# Cập nhật package list
sudo apt update

# Cài đặt các package cần thiết
sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release

# Thêm Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Thêm Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Cài đặt Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Thêm user vào docker group
sudo usermod -aG docker $USER

# Khởi động Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Trên Windows
1. Tải Docker Desktop từ [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Cài đặt và khởi động Docker Desktop
3. Đảm bảo WSL2 được kích hoạt

### Trên macOS
1. Tải Docker Desktop từ [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Cài đặt và khởi động Docker Desktop

## II. Tạo Docker Image PostgreSQL vs pgvector

### Phương án 1: Sử dụng Image có sẵn (Khuyến nghị)

1. Cài PgAdmin (Quản lý database)
```bash
# pull image PgAdmin có sẵn
docker pull dpage/pgadmin4

# Chạy PgAdmin
docker run --name pgadmin-container \
  -p 5050:80 \
  -e PGADMIN_DEFAULT_EMAIL=user@domain.com \
  -e PGADMIN_DEFAULT_PASSWORD=password \
  -d dpage/pgadmin4
```
2. Cài Pgvector (hỗ trợ dữ liệu vector)

```bash
# Pull image PostgreSQL với pgvector đã được build sẵn
docker pull pgvector/pgvector:pg16

# Tạo volume cho database
docker volume create pgvector-data

# Hoặc sử dụng image từ Harbor registry (nếu có)
docker pull harbor.cmc-u.edu.vn/aiagentdhs/pgvector:v1.0-pg16
```
```bash
# Chạy PostgreSQL container
docker run --name pgvector-container \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -v pgvector-data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
```
### Phương án 2: Build Image tùy chỉnh

Tạo file `Dockerfile.pgvector`:

```dockerfile
# Sử dụng PostgreSQL 16 làm base image
FROM postgres:16

# Cài đặt các package cần thiết để build pgvector
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

# Clone và build pgvector
RUN git clone --branch v0.2.2 https://github.com/pgvector/pgvector.git /tmp/pgvector
WORKDIR /tmp/pgvector
RUN make
RUN make install

# Tạo thư mục cho init scripts
RUN mkdir -p /docker-entrypoint-initdb.d

# Copy init script để tự động tạo extension
COPY init-pgvector.sql /docker-entrypoint-initdb.d/

# Cleanup
RUN rm -rf /tmp/pgvector
```

Tạo file `init-pgvector.sql`:

```sql
-- Tạo extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;
```

Build image:

```bash
# Build image
docker build -f Dockerfile.pgvector -t custom-pgvector:pg16 .

# Tag image cho Harbor registry (nếu cần)
docker tag custom-pgvector:pg16 harbor.cmc-u.edu.vn/aiagentdhs/pgvector:v1.0-pg16
```


## III. Khởi Động và Kiểm Tra

### Kiểm tra IP

```bash
# Lấy IP của PostgreSQL container
docker inspect --format "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" pgvector-container
```

### Kiểm tra logs

```bash
# Xem logs PostgreSQL
docker logs pgvector-container

# Xem logs PgAdmin
docker logs pgadmin-container

# Theo dõi logs real-time
docker logs -f pgvector-container
```

## IV. Kết Nối và Sử Dụng

### Từ ứng dụng Python

Cập nhật file `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_agent
DB_USER=postgres
DB_PASSWORD=newpassword
```

### Từ PgAdmin

1. Truy cập: `http://localhost:5050`
2. Đăng nhập:
    - Email: `user@domain.com`
    - Password: `password`
3. Thêm server mới:
   - Host: `pgvector-container`
   - Port: `5432`
   - Database: `ai_agent`
   - Username: `postgres`
   - Password: `newpassword`

### Từ command line

```bash
# Kết nối trực tiếp
docker exec -it pgvector-container psql -U postgres -d ai_agent

# Hoặc từ host (nếu có psql client)
psql -h localhost -p 5432 -U postgres -d ai_agent
```

## V. Khắc Phục Sự Cố

### Container không khởi động

```bash
# Kiểm tra logs chi tiết
docker logs pgvector-container

# Kiểm tra disk space
df -h

# Kiểm tra port conflict
netstat -tulpn | grep 5432
```

### Lỗi permission

```bash
# Fix quyền cho volume
sudo chown -R 999:999 ./pgdata
sudo chmod -R 755 ./pgdata
```

### Lỗi extension không tìm thấy

```bash
# Kiểm tra extension có sẵn
docker exec -it pgvector-container psql -U postgres -d ai_agent -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"

# Tạo extension thủ công
docker exec -it pgvector-container psql -U postgres -d ai_agent -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Reset database

```bash
# Dừng services
docker-compose -f docker-compose.pgvector.yml down

# Xóa volume (CẢNH BÁO: Mất hết dữ liệu)
docker volume rm $(docker volume ls -q | grep pgdata)

# Khởi động lại
docker-compose -f docker-compose.pgvector.yml up -d
```

## VI. Sao Lưu và Phục Hồi

### Sao lưu database

```bash
# Tạo backup
docker exec -t pgvector-container pg_dump -U postgres -d ai_agent > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup với compression
docker exec -t pgvector-container pg_dump -U postgres -d ai_agent | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Backup toàn bộ cluster
docker exec -t pgvector-container pg_dumpall -U postgres > full_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Phục hồi database

```bash
# Phục hồi từ file SQL
docker exec -i pgvector-container psql -U postgres -d ai_agent < backup_20241201_120000.sql

# Phục hồi từ file nén
gunzip -c backup_20241201_120000.sql.gz | docker exec -i pgvector-container psql -U postgres -d ai_agent
```

### Backup tự động

Tạo script `backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="ai_agent_backup_$DATE.sql"

mkdir -p $BACKUP_DIR

echo "Starting backup at $(date)"
docker exec -t pgvector-container pg_dump -U postgres -d ai_agent > "$BACKUP_DIR/$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"
    # Compress backup
    gzip "$BACKUP_DIR/$BACKUP_FILE"
    echo "Backup compressed: $BACKUP_FILE.gz"
    
    # Xóa backup cũ hơn 7 ngày
    find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
    echo "Old backups cleaned up"
else
    echo "Backup failed!"
    exit 1
fi
```

Thêm vào crontab để backup hàng ngày:

```bash
# Mở crontab
crontab -e

# Thêm dòng sau để backup lúc 2:00 AM hàng ngày
0 2 * * * /path/to/backup.sh >> /var/log/pgvector_backup.log 2>&1
```

## Tích Hợp với AI Agent Backend

Để tích hợp với dự án AI Agent Backend hiện tại:

1. **Cập nhật docker-compose.yml chính:**

```yaml
version: '3.8'

services:
  aiagent_backend:
    # ... existing config ...
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - DB_HOST=pgvector-container
      - DB_PORT=5432
      - DB_NAME=ai_agent
      - DB_USER=postgres
      - DB_PASSWORD=newpassword

  postgres:
    container_name: pgvector-container
    image: pgvector/pgvector:pg16
    restart: unless-stopped
    environment:
      POSTGRES_DB: ai_agent
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: newpassword
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - ai-agent-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d ai_agent"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  ai-agent-network:
    driver: bridge

volumes:
  pgdata:
```

2. **Khởi động toàn bộ hệ thống:**

```bash
docker-compose up -d
```

3. **Kiểm tra kết nối:**

```bash
# Kiểm tra backend có kết nối được database không
docker logs aiagent_backend | grep -i "database\|connection"

# Test API
curl http://localhost:8000/docs
```

## Kết Luận

Với hướng dẫn này, bạn đã có thể:

- ✅ Cài đặt Docker và Docker Compose
- ✅ Tạo PostgreSQL container vs pgvector extension
- ✅ Cấu hình PgAdmin để quản lý database
- ✅ Thiết lập backup và restore
- ✅ Tích hợp với AI Agent Backend

Hệ thống PostgreSQL với pgvector giờ đây sẵn sàng để lưu trữ và tìm kiếm vector embeddings cho ứng dụng RAG của bạn.

---

**Lưu ý quan trọng:**
- Luôn backup dữ liệu trước khi thực hiện thay đổi lớn
- Monitor disk space và performance
- Cập nhật password mặc định trong production
- Sử dụng SSL/TLS cho kết nối database trong môi trường production
