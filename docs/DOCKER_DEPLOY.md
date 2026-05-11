# Hướng dẫn triển khai AI Agent Chatbot Backend với Docker

Tài liệu này hướng dẫn cách triển khai backend của AI Agent Chatbot sử dụng Docker và Docker Compose.

## Điều kiện tiên quyết

*   **Docker Engine** và **Docker Compose:** Đã được cài đặt trên máy của bạn. Tải và cài đặt Docker Desktop (bao gồm cả Compose) từ [trang chủ Docker](https://www.docker.com/products/docker-desktop/).
*   **Mã nguồn dự án:** Đã có sẵn trên máy của bạn.

## Cấu hình Môi trường

1.  **Tạo file `.env`:**
    *   Trong thư mục gốc của dự án, bạn sẽ thấy file `env.docker`. Đây là file mẫu chứa các biến môi trường cần thiết.
    *   **Quan trọng:** Hãy tạo một bản sao của file này và đặt tên là `.env`.
      ```bash
      # Trên Linux/macOS
      cp .env.docker .env

      # Trên Windows (Command Prompt)
      copy .env.docker .env

      # Trên Windows (PowerShell)
      Copy-Item .env.docker .env
      ```
    *   **Chỉnh sửa file `.env`:** Mở file `.env` vừa tạo và cập nhật các giá trị cho phù hợp với môi trường của bạn, đặc biệt là:
        *   `DB_PASSWORD`: Mật khẩu cho user `postgres` của database.
        *   `OPENAI_API_KEY`, `GEMINI_API_KEY`, v.v.: Các API key cho dịch vụ LLM bạn muốn sử dụng.
    *   **Lưu ý:** File `.env` chứa thông tin nhạy cảm. **Không bao giờ commit file này vào Git repository.** Đảm bảo file `.env` đã được thêm vào `.gitignore` của bạn.

## Tạo Docker Volume và Network

Trước khi triển khai hệ thống, bạn cần tạo các Docker volumes và networks cần thiết:

1. **Tạo Docker volume cho việc chia sẻ uploads:**
   ```bash
   docker volume create shared_uploads
   ```
   Volume này sẽ được sử dụng để chia sẻ dữ liệu giữa các container, đặc biệt là giữa Node.js backend và Python backend.

2. **Kiểm tra mạng Docker (chỉ với Phương án 2):**
   ```bash
   docker network create aiagentsystem_ai-agent-network
   ```
   Mạng này sẽ cho phép các container giao tiếp với nhau thông qua tên container.

## Các Phương án Triển khai

Có hai cách chính để triển khai ứng dụng:

### Phương án 1: Triển khai Full Stack (Khuyên dùng cho thiết lập mới)

Phương án này sẽ sử dụng `docker-compose.full.yml` để khởi tạo và chạy đồng thời các container sau:
*   `aiagent_backend`: Container chứa ứng dụng backend.
*   `pgvector-container`: Container chứa database PostgreSQL với extension pgvector.
*   `pgadmin-container`: Container chứa công cụ quản lý database PgAdmin (truy cập qua trình duyệt).

**Các bước thực hiện:**

1.  **Mở Terminal hoặc PowerShell** trong thư mục gốc của dự án.
2.  **Tạo shared_uploads volume nếu chưa có:**
    ```bash
    docker volume create shared_uploads
    ```
3.  **Chạy script triển khai:**
    *   **Trên Linux/macOS:**
      ```bash
      chmod +x deploy_full.sh
      ./deploy_full.sh
      ```
    *   **Trên Windows (PowerShell):**
      ```powershell
      .\deploy_full.ps1
      ```
4.  **Quá trình:** Script sẽ tự động:
    *   Tạo các thư mục `uploadfiles` và `temp` nếu chưa có.
    *   Dừng và xóa các container cũ (nếu có) được định nghĩa trong `docker-compose.full.yml`.
    *   Build (hoặc rebuild) image `aiagent_backend` nếu có thay đổi.
    *   Khởi động tất cả các container trong chế độ detached (`-d`).
    *   Hiển thị trạng thái các container đang chạy.
5.  **Kết nối PostgreSQL với mạng Node.js (nếu cần):**
    ```bash
    docker network connect aiagentsystem_ai-agent-network pgvector-container
    ```
    Lệnh này đảm bảo container pgvector-container có thể giao tiếp với Node.js backend.
6.  **Truy cập:**
    *   **API Backend:** `http://localhost:8000/docs` (Giao diện Swagger UI)
    *   **PgAdmin:** `http://localhost:5050`
        *   *Email đăng nhập mặc định:* `admin@example.com`
        *   *Mật khẩu mặc định:* `admin` (Có thể thay đổi trong `docker-compose.full.yml`)
    *   **Database (từ PgAdmin hoặc ứng dụng khác):**
        *   *Host:* `pgvector-container` (nếu kết nối từ container khác trong cùng mạng) hoặc IP của container `pgvector-container`.
        *   *Port:* `5432`
        *   *Database Name:* `ai_agent`
        *   *Username:* `postgres`
        *   *Password:* Mật khẩu bạn đã đặt trong file `.env` (`DB_PASSWORD`).

### Phương án 2: Chỉ triển khai Backend (Khi đã có Database)

Phương án này chỉ khởi động container `aiagent_backend`. Bạn cần đảm bảo đã có một instance PostgreSQL (phiên bản tương thích với pgvector) đang chạy và có thể truy cập được từ container backend.

**Yêu cầu:**

*   Database PostgreSQL đã được cài đặt và đang chạy (có thể là container Docker khác hoặc chạy trực tiếp trên máy/server khác).
*   Extension `pgvector` đã được cài đặt và kích hoạt trong database đó.
*   Đã cấu hình file `.env` với thông tin kết nối chính xác đến database hiện có (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`). `DB_HOST` phải là địa chỉ mà container backend có thể phân giải được (ví dụ: tên container khác trong cùng mạng Docker, IP nội bộ, hoặc tên miền).

**Các bước thực hiện:**

1.  **Đảm bảo mạng Docker tồn tại (nếu DB là container khác):** Nếu database của bạn đang chạy trong một container Docker khác (ví dụ: tên là `pgvector-container`), hãy đảm bảo cả container đó và container backend sắp tạo đều được kết nối vào cùng một mạng Docker (ví dụ: `aiagentsystem_ai-agent-network`).
    ```bash
    # Tạo mạng nếu chưa có
    docker network create aiagentsystem_ai-agent-network

    # Kết nối container DB hiện có vào mạng
    docker network connect aiagentsystem_ai-agent-network pgvector-container
    ```
2.  **Tạo volume shared_uploads:**
    ```bash
    docker volume create shared_uploads
    ```
3.  **Build image backend (nếu chưa có hoặc có thay đổi):**
    ```bash
    docker build -t aiagent_backend .
    ```
4.  **Chạy container backend:**
    *   **Trên Linux/macOS:**
      ```bash
      # Dừng và xóa container cũ nếu đang chạy
      docker stop aiagent_backend || true
      docker rm aiagent_backend || true

      docker run -d --name aiagent_backend \
        --network aiagentsystem_ai-agent-network \
        -p 8000:8000 \
        -v shared_uploads:/app/uploads \
        -v "$(pwd)/uploadfiles:/app/uploadfiles" \
        -v "$(pwd)/temp:/app/temp" \
        --env-file .env \
        aiagent_backend
      ```
    *   **Trên Windows (PowerShell):**
      ```powershell
      # Dừng và xóa container cũ nếu đang chạy
      docker stop aiagent_backend; docker rm aiagent_backend

      docker run -d --name aiagent_backend `
        --network aiagentsystem_ai-agent-network `
        -p 8000:8000 `
        -v shared_uploads:/app/uploads `
        -v "${PWD}/uploadfiles:/app/uploadfiles" `
        -v "${PWD}/temp:/app/temp" `
        --env-file .env `
        aiagent_backend
      ```
    *   **Giải thích lệnh:**
        *   `-d`: Chạy container ở chế độ detached (chạy nền).
        *   `--name aiagent_backend`: Đặt tên cho container.
        *   `--network aiagentsystem_ai-agent-network`: Kết nối container vào mạng để có thể giao tiếp với container database.
        *   `-p 8000:8000`: Ánh xạ cổng 8000 của máy host vào cổng 8000 của container.
        *   `-v shared_uploads:/app/uploads`: Sử dụng volume Docker để chia sẻ dữ liệu với Node.js backend.
        *   `-v ...:/app/uploadfiles`: Mount thư mục `uploadfiles` trên máy host vào `/app/uploadfiles` trong container.
        *   `-v ...:/app/temp`: Mount thư mục `temp` trên máy host vào `/app/temp` trong container.
        *   `--env-file .env`: Đọc các biến môi trường từ file `.env`.
        *   `aiagent_backend`: Tên image Docker để chạy.
5.  **Truy cập API Backend:** `http://localhost:8000/docs`

## Vấn đề thường gặp và cách khắc phục

### 1. Không tìm thấy file đã upload từ Node.js backend
- **Vấn đề:** Python backend không thể tìm thấy file đã được upload qua Node.js backend.
- **Nguyên nhân:** Hai container đang sử dụng các thư mục uploads khác nhau.
- **Giải pháp:** 
  - Sử dụng Docker volume `shared_uploads` cho cả hai container.
  - Đảm bảo Node.js backend cũng sử dụng volume này trong docker-compose:
    ```yaml
    volumes:
      - shared_uploads:/app/uploads
    ```
  - Thêm vào cuối file docker-compose của Node.js backend:
    ```yaml
    volumes:
      shared_uploads:
        external: true
    ```

### 2. Không kết nối được với database
- **Vấn đề:** Lỗi "could not translate host name pgvector-container to address."
- **Nguyên nhân:** Các container không nằm trong cùng mạng Docker.
- **Giải pháp:**
  ```bash
  docker network connect aiagentsystem_ai-agent-network pgvector-container
  ```

### 3. Vấn đề đường dẫn trong Docker trên Windows
- **Vấn đề:** Docker không nhận diện đường dẫn Windows chính xác.
- **Giải pháp:** 
  - Sử dụng Docker volume thay vì bind mount.
  - Hoặc sử dụng đường dẫn Unix-style: `/c/Users/...`
  - Hoặc đường dẫn Windows với dấu gạch chéo: `C:/Users/...`

## Kiểm tra và Gỡ lỗi

*   **Kiểm tra trạng thái container:**
    ```bash
    docker ps
    ```
    (Xem các container đang chạy)
    ```bash
    docker ps -a
    ```
    (Xem tất cả các container, kể cả đã dừng)
*   **Xem logs của container:** (Rất hữu ích để gỡ lỗi)
    ```bash
    # Xem logs của backend
    docker logs aiagent_backend

    # Xem logs của database (nếu dùng full stack)
    docker logs pgvector-container

    # Theo dõi logs liên tục (thêm -f)
    docker logs -f aiagent_backend
    ```
*   **Kiểm tra kết nối mạng:**
    ```bash
    # Kiểm tra các mạng đang có
    docker network ls
    
    # Kiểm tra chi tiết một mạng
    docker network inspect aiagentsystem_ai-agent-network
    ```
*   **Kiểm tra volume:**
    ```bash
    # Liệt kê volumes
    docker volume ls
    
    # Kiểm tra chi tiết volume
    docker volume inspect shared_uploads
    ```
*   **Kiểm tra mounts của container:**
    ```bash
    docker inspect aiagent_backend -f "{{json .Mounts}}"
    ```
*   **Truy cập vào bên trong container (để kiểm tra):**
    ```bash
    docker exec -it aiagent_backend bash
    ```
    (Sau đó bạn có thể dùng các lệnh Linux bên trong container)

## Dừng ứng dụng

*   **Nếu bạn đã triển khai Full Stack (Phương án 1):**
    ```bash
    docker compose -f docker-compose.full.yml down
    ```
    Lệnh này sẽ dừng và xóa các container, mạng được tạo bởi file compose này. Volume `pgdata` và `shared_uploads` sẽ không bị xóa mặc định, giúp giữ lại dữ liệu.
*   **Nếu bạn chỉ triển khai Backend (Phương án 2):**
    ```bash
    docker stop aiagent_backend
    docker rm aiagent_backend
    ```

---
Chúc bạn triển khai thành công! 