# Hệ thống Chatbot AI Agent

## Giới thiệu

Hệ thống chatbot sử dụng LLM với khả năng xử lý tài liệu thông qua vector database. Hệ thống hỗ trợ nhiều định dạng tài liệu khác nhau bao gồm PDF, Word, và text với khả năng OCR cho tài liệu scan.

## Tính năng chính

- Trích xuất văn bản từ nhiều loại tài liệu (PDF, DOCX, TXT)
- OCR (Optical Character Recognition) cho tài liệu scan
- Chunking văn bản theo nhiều chiến lược
- Tạo vector embedding
- Hỗ trợ nhiều LLM khác nhau (OpenAI, Google, Cohere)
- API đầy đủ chức năng

## Semantic Chunking - Phương pháp Clustering

Dự án đã được bổ sung thêm phương pháp chunking sử dụng phân cụm (clustering) để tăng hiệu quả trong việc phân đoạn văn bản theo nội dung ngữ nghĩa.

### Các tính năng mới
- **Phân cụm phân cấp (Hierarchical Clustering)**: Sử dụng thuật toán AgglomerativeClustering để nhóm các câu có nội dung liên quan với nhau
- **Ma trận tương đồng toàn diện**: Phân tích mối quan hệ giữa tất cả các câu thay vì chỉ so sánh các câu liền kề
- **Tùy chọn bảo toàn thứ tự**: Có thể duy trì thứ tự ban đầu của văn bản hoặc sắp xếp lại theo cluster

### Cách sử dụng
Bạn có thể kích hoạt tính năng clustering bằng cách gọi API endpoint sau:

```bash
# Kích hoạt chức năng clustering
curl -X POST "http://localhost:8000/api/semantic/config" \
  -H "Content-Type: application/json" \
  -d '{
    "use_clustering": true,
    "preserve_order": true,
    "threshold": 0.5
  }'
```

### Tham số
- `use_clustering`: `true/false` - Kích hoạt phương pháp phân cụm
- `preserve_order`: `true/false` - Duy trì thứ tự ban đầu của văn bản
- `threshold`: `0.0 - 1.0` - Ngưỡng tương đồng để gom nhóm các câu (giá trị càng cao, yêu cầu càng tương đồng)

### Theo dõi quá trình phân cụm (clustering)
Hệ thống được tích hợp logging chi tiết cho phương pháp phân cụm. Bạn có thể theo dõi quá trình clustering thông qua:

1. **Log trong console**: Khi khởi động ứng dụng với `uvicorn main:app --reload`
2. **File log**: Tất cả log sẽ được lưu trong file `app.log` tại thư mục gốc của dự án
3. **Chi tiết log bao gồm**:
   - Số lượng clusters được tạo ra
   - Phân bố câu trong mỗi cluster
   - Quá trình tạo chunk từ các clusters
   - Thông số của mỗi chunk (kích thước, cluster id)

Ví dụ log:
```
2023-10-15 10:25:30 - chunking.semantic_chunker - INFO - Bắt đầu quá trình phân đoạn với phương pháp CLUSTERING
2023-10-15 10:25:31 - chunking.semantic_chunker - INFO - Tổng số câu sau khi lọc: 45
2023-10-15 10:25:33 - chunking.semantic_chunker - INFO - Kết quả phân cụm: 8 clusters từ 45 câu
2023-10-15 10:25:33 - chunking.semantic_chunker - INFO - Chi tiết phân cụm: {0: 12, 1: 8, 2: 5, 3: 6, 4: 4, 5: 3, 6: 5, 7: 2}
```

### Ưu điểm so với phương pháp truyền thống
1. Nhận diện được các câu có liên quan ngữ nghĩa với nhau ngay cả khi chúng không nằm cạnh nhau
2. Tạo ra các đoạn văn có tính liên kết cao hơn về mặt ngữ nghĩa
3. Phù hợp với nội dung có cấu trúc phức tạp, không tuân theo trình tự tuyến tính

## Cài đặt

### Yêu cầu

- Python 3.8+
- PostgreSQL với pgvector
- Tesseract OCR
- Poppler

### Cài đặt OCR

Để cài đặt và cấu hình hệ thống OCR, vui lòng tham khảo tài liệu chi tiết tại:
[Hướng dẫn cài đặt OCR](OCR_SETUP.md)

### Cài đặt thư viện Python

```bash
pip install -r requirements.txt
```

### Cấu hình

Sao chép file `.env.example` thành `.env` và cập nhật các biến môi trường cần thiết.

## Sử dụng

### Khởi động server

```bash
uvicorn main:app --reload
```

### Các API chính

- `/upload/chunking/`: Tải lên và xử lý tài liệu
- `/chunking/config/`: Quản lý cấu hình chunking
- `/ocr/config/`: Kiểm tra cấu hình OCR hiện tại

## Tài liệu liên quan

- [Hướng dẫn cài đặt OCR](OCR_SETUP.md)
- [Tài liệu API](/docs/api.md) 