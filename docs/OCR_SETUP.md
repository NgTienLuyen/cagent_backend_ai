# HƯỚNG DẪN CÀI ĐẶT VÀ CẤU HÌNH OCR CHI TIẾT

## 1. Cài đặt Tesseract OCR

### 1.1. Tải Tesseract OCR

1. Truy cập trang web chính thức của Tesseract OCR cho Windows: https://github.com/UB-Mannheim/tesseract/wiki
2. Tìm đến phần "Download", cuộn xuống phần "Latest installer" để tải phiên bản mới nhất
3. Chọn phiên bản phù hợp với hệ điều hành của bạn:
   - `tesseract-ocr-w64-setup-vX.X.X.exe` cho Windows 64-bit (khuyến nghị)
   - `tesseract-ocr-w32-setup-vX.X.X.exe` cho Windows 32-bit
4. Nhấp vào liên kết để tải xuống file cài đặt

### 1.2. Cài đặt Tesseract OCR

1. Tìm file cài đặt đã tải xuống (thường ở thư mục Downloads)
2. Nhấp chuột phải vào file và chọn "Run as administrator" (Chạy với quyền quản trị)
3. Trong cửa sổ cài đặt, nhấp "Next" để tiếp tục
4. Chọn đường dẫn cài đặt: `C:\Program Files\Tesseract-OCR` (đây là đường dẫn mặc định)
5. Ở màn hình "Select Components":
   - Đảm bảo "Tesseract OCR" đã được chọn
   - Đánh dấu vào "Additional language data (download)"
   - Nhấp "Next"
6. Ở màn hình "Additional language data":
   - Tìm và đánh dấu vào "Vietnamese" hoặc "vie" trong danh sách
   - Nếu cần nhiều ngôn ngữ khác, bạn có thể chọn thêm
   - Nhấp "Next"
7. Ở màn hình "Select Additional Tasks":
   - Đánh dấu vào "Add to PATH" để thêm Tesseract vào biến môi trường
   - Nhấp "Next"
8. Nhấp "Install" để bắt đầu cài đặt
9. Đợi quá trình cài đặt hoàn tất và nhấp "Finish"

### 1.3. Kiểm tra cài đặt Tesseract

1. Mở Command Prompt:
   - Nhấn tổ hợp phím `Windows + R`
   - Nhập `cmd` và nhấn Enter
2. Nhập lệnh sau để kiểm tra phiên bản Tesseract:
   ```
   "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   ```
   Bạn sẽ thấy thông tin phiên bản Tesseract
3. Kiểm tra các ngôn ngữ đã cài đặt:
   ```
   "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
   ```
   Đảm bảo "vie" xuất hiện trong danh sách

### 1.4. Cài đặt thủ công ngôn ngữ tiếng Việt (nếu cần)

Nếu bạn không thấy "vie" trong danh sách ngôn ngữ, hãy cài đặt thủ công:

1. Truy cập: https://github.com/tesseract-ocr/tessdata/
2. Tìm file `vie.traineddata` và nhấp vào nó
3. Nhấp vào nút "Download" để tải file về máy
4. Sao chép file đã tải về vào thư mục: `C:\Program Files\Tesseract-OCR\tessdata\`
5. Kiểm tra lại danh sách ngôn ngữ như ở bước 1.3 để đảm bảo "vie" đã xuất hiện

## 2. Cài đặt Poppler

### 2.1. Tải Poppler

1. Truy cập trang GitHub của Poppler cho Windows: https://github.com/oschwartz10612/poppler-windows/releases/
2. Cuộn xuống và tìm phiên bản mới nhất (thường là mục đầu tiên có nhãn "Latest")
3. Tải xuống file .zip phù hợp (ưu tiên file Release-xx.xx.x-0.zip)

### 2.2. Cài đặt Poppler

1. Tìm file .zip đã tải xuống (thường ở thư mục Downloads)
2. Nhấp chuột phải vào file và chọn "Extract All..." (Giải nén tất cả...)
3. Trong hộp thoại, thay đổi đường dẫn thành `C:\` (hoặc đường dẫn bạn muốn)
4. Nhấp "Extract" để giải nén
5. Đổi tên thư mục được giải nén (thường có dạng `Release-xx.xx.x-0` hoặc `poppler-xx.xx.x`) thành `poppler`
6. Đảm bảo có thư mục `bin` trong `C:\poppler\` và chứa các file như `pdfinfo.exe`, `pdftoppm.exe`, v.v.

### 2.3. Kiểm tra cài đặt Poppler

1. Mở Command Prompt:
   - Nhấn tổ hợp phím `Windows + R`
   - Nhập `cmd` và nhấn Enter
2. Nhập lệnh sau để kiểm tra phiên bản Poppler:
   ```
   "C:\poppler\bin\pdfinfo.exe" -v
   ```
   Bạn sẽ thấy thông tin phiên bản Poppler

## 3. Cấu hình trong dự án

### 3.1. Tạo và cấu hình file .env

1. Mở thư mục gốc của dự án trong File Explorer
2. Kiểm tra xem đã có file `.env` chưa:
   - Nếu chưa có, tạo một file văn bản mới và đặt tên là `.env` (đảm bảo loại bỏ phần mở rộng .txt)
   - Nếu đã có, hãy mở file đó với trình soạn thảo văn bản như Notepad, Visual Studio Code, v.v.
3. Thêm hoặc cập nhật các dòng sau vào file `.env`:

```
# Tesseract OCR configuration
TESSERACT_CMD_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\poppler\bin
OCR_LANGUAGE=vie
OCR_DPI=600
OCR_CONFIG=--psm 1 --oem 3 -c textord_min_linesize=3.0 -c preserve_interword_spaces=1
```

4. Lưu file `.env`

### 3.2. Giải thích chi tiết các tham số OCR

1. **TESSERACT_CMD_PATH**:
   - Đây là đường dẫn đến file thực thi của Tesseract OCR
   - Đảm bảo đường dẫn chính xác và không có khoảng trắng thừa
   - Ví dụ: `TESSERACT_CMD_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe`

2. **POPPLER_PATH**:
   - Đây là đường dẫn đến thư mục bin của Poppler
   - Đảm bảo đường dẫn chính xác và không có khoảng trắng thừa
   - Ví dụ: `POPPLER_PATH=C:\poppler\bin`

3. **OCR_LANGUAGE**:
   - Xác định ngôn ngữ mà Tesseract sẽ sử dụng để nhận dạng văn bản
   - `vie`: Tiếng Việt
   - `eng`: Tiếng Anh
   - Có thể kết hợp nhiều ngôn ngữ bằng dấu +, ví dụ: `vie+eng`

4. **OCR_DPI**:
   - Độ phân giải khi chuyển đổi PDF sang ảnh để OCR
   - Giá trị cao hơn = độ chính xác cao hơn nhưng xử lý chậm hơn
   - Giá trị đề xuất: 600 (mặc định là 300)
   - Phạm vi khuyến nghị: 300-900

5. **OCR_CONFIG**:
   - Các tham số cấu hình chi tiết cho Tesseract OCR
   - `--psm 1`: Page Segmentation Mode = Tự động phân đoạn trang với định hướng
   - `--oem 3`: OCR Engine Mode = Kết hợp cả LSTM và Legacy engine
   - `-c textord_min_linesize=3.0`: Cải thiện nhận dạng chữ nhỏ
   - `-c preserve_interword_spaces=1`: Giữ nguyên khoảng cách giữa các từ

### 3.3. Thêm các tùy chọn nâng cao vào file .env (tùy chọn)

Nếu bạn muốn sử dụng Google Vision API để OCR (cho kết quả tốt hơn), thêm dòng sau:

```
# Google Vision API configuration
GOOGLE_APPLICATION_CREDENTIALS=đường/dẫn/đến/file-credentials.json
```

## 4. Cài đặt các thư viện Python cần thiết

### 4.1. Cài đặt các gói Python

1. Mở Command Prompt hoặc PowerShell
2. Di chuyển đến thư mục dự án:
   ```
   cd đường\dẫn\đến\thư_mục_dự_án
   ```
3. Cài đặt các thư viện cần thiết:
   ```
   pip install pytesseract pdf2image python-dotenv pillow opencv-python
   ```

### 4.2. Kiểm tra cài đặt thư viện

1. Mở Python trong Command Prompt:
   ```
   python
   ```
2. Thử import các thư viện:
   ```python
   import pytesseract
   import pdf2image
   import dotenv
   import cv2
   import PIL
   print("Tất cả thư viện đã được cài đặt thành công!")
   exit()
   ```

## 5. Sử dụng OCR trong dự án

### 5.1. Khởi động server FastAPI

1. Mở Command Prompt hoặc PowerShell
2. Di chuyển đến thư mục dự án:
   ```
   cd đường\dẫn\đến\thư_mục_dự_án
   ```
3. Khởi động server FastAPI:
   ```
   uvicorn main:app --reload
   ```

### 5.2. Sử dụng API OCR

#### 5.2.1. Kiểm tra cấu hình OCR hiện tại

1. Mở trình duyệt web
2. Truy cập: `http://localhost:8000/ocr/config/`
3. Bạn sẽ thấy thông tin cấu hình OCR hiện tại dưới dạng JSON

#### 5.2.2. Kiểm tra Tesseract đã hoạt động

1. Truy cập: `http://localhost:8000/ocr/check_tesseract/`
2. Kết quả trả về sẽ cho biết Tesseract đã được cài đặt đúng hay chưa

#### 5.2.3. Xử lý tài liệu với OCR

1. Truy cập: `http://localhost:8000/docs`
2. Tìm endpoint `/upload/chunking/`
3. Nhấp vào "Try it out"
4. Nhập các tham số:
   - `document_id`: ID của tài liệu cần xử lý
   - `ocr_method`: Chọn phương pháp OCR (`tesseract`, `google_vision` hoặc `auto`)
5. Nhấp "Execute" để thực hiện

## 6. Khắc phục sự cố chi tiết

### 6.1. Lỗi "Unable to get page count. Is poppler installed and in PATH?"

**Nguyên nhân**: Poppler không được cài đặt đúng hoặc đường dẫn không chính xác.

**Cách khắc phục**:
1. Kiểm tra đường dẫn Poppler trong file `.env`: `POPPLER_PATH=C:\poppler\bin`
2. Đảm bảo thư mục `C:\poppler\bin` tồn tại và chứa các file như `pdfinfo.exe`, `pdftoppm.exe`
3. Thử giải nén lại Poppler vào đúng thư mục
4. Khởi động lại server sau khi sửa

### 6.2. Lỗi "Error opening data file C:\\Program Files\\Tesseract-OCR/tessdata/vie.traineddata"

**Nguyên nhân**: Không tìm thấy file dữ liệu ngôn ngữ tiếng Việt.

**Cách khắc phục**:
1. Kiểm tra thư mục `C:\Program Files\Tesseract-OCR\tessdata\` có chứa file `vie.traineddata` không
2. Nếu không, tải file từ: https://github.com/tesseract-ocr/tessdata/raw/main/vie.traineddata
3. Đặt file vào thư mục `C:\Program Files\Tesseract-OCR\tessdata\`
4. Khởi động lại server

### 6.3. Lỗi khi đọc file .env

**Nguyên nhân**: File .env không được đọc đúng cách hoặc định dạng không đúng.

**Cách khắc phục**:
1. Đảm bảo file `.env` ở đúng thư mục gốc của dự án
2. Kiểm tra định dạng file không có "set" hoặc "export" ở đầu mỗi dòng:
   ```
   # Thử các giá trị khác nhau:
   --psm 1  # Tự động phân đoạn trang với hướng
   --psm 3  # Tự động phân đoạn trang không có hướng (thường tốt nhất)
   --psm 6  # Xem là một khối văn bản duy nhất
   ```
3. Thử sử dụng Google Vision API cho kết quả tốt hơn (cấu hình `ocr_method=google_vision`)

## 7. Tối ưu OCR cho từng loại tài liệu

### 7.1. Tài liệu tiếng Việt thông thường

Cấu hình khuyến nghị:
```
OCR_LANGUAGE=vie
OCR_DPI=600
OCR_CONFIG=--psm 3 --oem 3 -c preserve_interword_spaces=1
```

### 7.2. Tài liệu có nhiều bảng và cột

Cấu hình khuyến nghị:
```
OCR_LANGUAGE=vie
OCR_DPI=600
OCR_CONFIG=--psm 1 --oem 3 -c preserve_interword_spaces=1
```

### 7.3. Tài liệu có chữ nhỏ hoặc mờ

Cấu hình khuyến nghị:
```
OCR_LANGUAGE=vie
OCR_DPI=900
OCR_CONFIG=--psm 3 --oem 3 -c textord_min_linesize=2.5
```

## 8. Tham khảo thêm

- [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/)
- [Tesseract GitHub Repository](https://github.com/tesseract-ocr/tesseract)
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows)
- [PyTesseract GitHub](https://github.com/madmaze/pytesseract)
- [PDF2Image GitHub](https://github.com/Belval/pdf2image)
- [Tesseract OCR Parameters](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html) 