# semantic_chunking_api.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body
import uuid
from pgvector.asyncpg import register_vector
from chunking.semantic_chunker import SemanticChunker, ChunkingConfig
from database.db_connection import get_pg_connection, return_pg_connection
import pdfplumber
from io import BytesIO
import pandas as pd
import docx
from typing import Optional, List, Dict, Any
import json
import logging
import os
import pytesseract
from pdf2image import convert_from_path
import subprocess
from pgvector.psycopg2 import register_vector as register_vector_sync # Đổi tên để tránh nhầm lẫn

# Thêm import underthesea
try:
    from underthesea import word_tokenize, sent_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    logging.warning("Thư viện underthesea không được tìm thấy. API sẽ sử dụng NLTK cho tiếng Việt.")

# Thêm cho Google Cloud Vision API
from dotenv import load_dotenv

# Tải lại biến môi trường từ file .env
load_dotenv(override=True)

# In ra giá trị biến môi trường OCR để debug
print("==== OCR CONFIG VALUES LOADED ====")
print(f"OCR_LANGUAGE: {os.getenv('OCR_LANGUAGE')}")
print(f"OCR_DPI: {os.getenv('OCR_DPI')}")
print(f"OCR_CONFIG: {os.getenv('OCR_CONFIG')}")
print("=================================")

router = APIRouter()  # Tạo router cho chunking

# Tải lại biến môi trường từ file .env mỗi khi module được import
load_dotenv(override=True)


# API để lấy cấu hình mặc định và các tùy chọn khả dụng
@router.get("/chunking/config/")
async def get_chunking_config():
    """Lấy cấu hình mặc định và các tùy chọn khả dụng cho chunking"""
    default_config = ChunkingConfig.get_default_config()

    return {
        "default_config": default_config.dict(),
        "available_options": {
            "embedding_types": ChunkingConfig.get_available_embedding_types(),
            "models": ChunkingConfig.get_available_models(),
            "languages": ChunkingConfig.get_available_languages(),
            "underthesea_support": UNDERTHESEA_AVAILABLE  # Thêm trạng thái hỗ trợ underthesea
        }
    }


# API để lưu cấu hình chunking của người dùng
@router.post("/chunking/config/save/")
async def save_chunking_config(config: ChunkingConfig, user_id: str = Query(...)):
    """Lưu cấu hình chunking của người dùng"""
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            # Kiểm tra xem người dùng đã có cấu hình chưa
            cursor.execute("""
                SELECT id FROM user_chunking_configs WHERE user_id = %s
            """, (user_id,))

            existing_config = cursor.fetchone()

            if existing_config:
                # Cập nhật cấu hình hiện có
                cursor.execute("""
                    UPDATE user_chunking_configs 
                    SET config = %s, updated_at = NOW()
                    WHERE user_id = %s
                """, (json.dumps(config.dict()), user_id))
            else:
                # Tạo cấu hình mới
                cursor.execute("""
                    INSERT INTO user_chunking_configs (user_id, config, created_at, updated_at)
                    VALUES (%s, %s, NOW(), NOW())
                """, (user_id, json.dumps(config.dict())))

            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


# API để lấy cấu hình chunking của người dùng
@router.get("/chunking/config/user/")
async def get_user_chunking_config(user_id: str = Query(...)):
    """Lấy cấu hình chunking của người dùng"""
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT config FROM user_chunking_configs WHERE user_id = %s
            """, (user_id,))

            config_row = cursor.fetchone()
            if config_row:
                config_dict = json.loads(config_row[0])
                return {"status": "success", "config": config_dict}
            else:
                # Trả về cấu hình mặc định nếu người dùng chưa có cấu hình
                return {"status": "success", "config": ChunkingConfig().dict(), "is_default": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user config: {str(e)}")
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


@router.post("/upload/chunking/")
async def upload_chunking(
        document_id: str = Query(..., description="UUID của tài liệu muốn chunking"),
        config: Optional[ChunkingConfig] = Body(None, description="Cấu hình chunking tùy chọn"),
        ocr_method: str = Query("tesseract", description="Phương pháp OCR: 'tesseract', 'google_vision', hoặc 'auto'")
):
    logger = logging.getLogger(__name__)
    logger.info(f"Nhận yêu cầu chunking cho document_id: {document_id}")
    
    if config and config.use_clustering:
        logger.info(f"Sử dụng phương pháp CLUSTERING với threshold={config.threshold}, preserve_order={config.preserve_order}")
    
    try:
        # Kiểm tra document_id có hợp lệ không
        document_uuid = uuid.UUID(document_id)  # Chuyển chuỗi thành UUID
        document = get_document_by_id(document_id=str(document_uuid))
        if not document:
            logger.error(f"Không tìm thấy tài liệu với ID: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found.")

        # Đọc nội dung file tài liệu
        file_path = document["document_link"]  # Lấy đường dẫn tài liệu từ bảng documents
        file_path = file_path.replace("\\\\", "/")
        logger.info(f"Đọc nội dung từ file: {file_path}")
        content = await read_file_content(file_path, ocr_method)

        # Thực hiện chunking văn bản với cấu hình được cung cấp hoặc mặc định
        chunks = await process_text(content, config)

        # Lưu các chunk vào bảng chunks
        logger.info(f"Lưu {len(chunks)} chunks vào cơ sở dữ liệu")
        await save_to_db_chunking(file_path, chunks, document_id, config)

        return {
            "document_id": document_id,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "config_used": config.dict() if config else ChunkingConfig().dict(),
            "ocr_method": ocr_method
        }
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chunking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# API để thử nghiệm chunking mà không lưu vào cơ sở dữ liệu
@router.post("/chunking/preview/")
async def preview_chunking(
        text: str = Body(..., description="Văn bản cần phân đoạn"),
        config: Optional[ChunkingConfig] = Body(None, description="Cấu hình chunking tùy chọn")
):
    logger = logging.getLogger(__name__)
    logger.info("Nhận yêu cầu preview chunking")
    
    # Thông báo về hỗ trợ underthesea nếu ngôn ngữ là tiếng Việt
    if config and config.language == "vietnamese":
        if UNDERTHESEA_AVAILABLE:
            logger.info("Sử dụng underthesea cho xử lý tiếng Việt")
        else:
            logger.warning("Ngôn ngữ tiếng Việt được chọn nhưng underthesea không khả dụng, sẽ sử dụng NLTK thay thế")
    
    if config and config.use_clustering:
        logger.info(f"Preview sử dụng phương pháp CLUSTERING với threshold={config.threshold}, preserve_order={config.preserve_order}")
    
    try:
        # Thực hiện chunking văn bản với cấu hình được cung cấp hoặc mặc định
        chunks = await process_text(text, config)
        logger.info(f"Preview chunking: tạo {len(chunks)} chunks")

        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "config_used": config.dict() if config else ChunkingConfig().dict(),
            "underthesea_support": UNDERTHESEA_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Lỗi trong quá trình preview chunking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Hàm xử lý OCR trực tuyến sử dụng Google Cloud Vision API
async def ocr_with_google_vision(pdf_path: str) -> str:
    """
    Sử dụng Google Cloud Vision API để OCR trên file PDF.
    Lưu ý: Cần cài đặt gói google-cloud-vision và có credentials JSON.
    """
    try:
        from google.cloud import vision
        from pdf2image import convert_from_path
        import io
        
        logger = logging.getLogger(__name__)
        logger.info(f"Bắt đầu OCR trực tuyến với Google Cloud Vision cho file {pdf_path}")
        
        # Lấy đường dẫn đến credentials JSON từ biến môi trường
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            logger.warning("Không tìm thấy GOOGLE_APPLICATION_CREDENTIALS trong biến môi trường")
            # Bạn có thể đặt đường dẫn mặc định ở đây nếu cần
            # credentials_path = "path/to/your/credentials.json"
        
        # Khởi tạo client Vision API
        client = vision.ImageAnnotatorClient()
        
        # Chuyển PDF thành ảnh
        images = convert_from_path(pdf_path)
        logger.info(f"Đã chuyển đổi PDF thành {len(images)} hình ảnh")
        
        all_texts = []
        
        # Xử lý từng trang
        for i, image in enumerate(images):
            logger.info(f"Đang xử lý OCR cho trang {i+1}/{len(images)}")
            
            # Chuyển ảnh thành byte stream
            byte_stream = io.BytesIO()
            image.save(byte_stream, format="JPEG")
            byte_stream.seek(0)
            content = byte_stream.read()
            
            # Tạo vision image và thực hiện OCR
            vision_image = vision.Image(content=content)
            response = client.document_text_detection(image=vision_image)
            
            # Kiểm tra lỗi
            if response.error.message:
                raise Exception(f"Google Vision API error: {response.error.message}")
            
            # Lấy văn bản
            text = response.full_text_annotation.text
            if text:
                all_texts.append(text)
            else:
                logger.warning(f"Không tìm thấy văn bản trên trang {i+1}")
        
        if not all_texts:
            logger.error("Không thể trích xuất văn bản nào từ PDF sử dụng Google Vision API")
            return ""
        
        combined_text = "\n\n".join(all_texts)
        logger.info(f"OCR hoàn tất, đã trích xuất {len(combined_text)} ký tự")
        return combined_text
        
    except ImportError as e:
        logger.error(f"Thiếu thư viện cho Google Cloud Vision: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cần cài đặt thư viện: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi khi sử dụng Google Cloud Vision OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi OCR trực tuyến: {str(e)}")


# Hàm xử lý OCR với Tesseract
async def ocr_with_tesseract(pdf_path: str) -> str:
    """
    Sử dụng Tesseract OCR để trích xuất văn bản từ PDF dạng scan.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Bắt đầu OCR với Tesseract cho file {pdf_path}")
        
        # In lại các giá trị biến môi trường để debug
        logger.info("==== OCR CONFIG VALUES ====")
        logger.info(f"OCR_LANGUAGE: {os.getenv('OCR_LANGUAGE')}")
        logger.info(f"OCR_DPI: {os.getenv('OCR_DPI')}")
        logger.info(f"OCR_CONFIG: {os.getenv('OCR_CONFIG')}")
        logger.info("==========================")
        
        # Kiểm tra nếu đang trong môi trường Docker
        docker_env = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        if docker_env:
            # Trong Docker, chỉ đơn giản sử dụng lệnh 'tesseract'
            logger.info("Đang chạy trong Docker, sử dụng lệnh 'tesseract'")
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'
        else:
            # Đặt đường dẫn Tesseract nếu cần - chỉ cho môi trường không phải Docker
            tesseract_path = os.getenv("TESSERACT_CMD_PATH")
            if tesseract_path:
                # Đảm bảo không có khoảng trắng thừa
                tesseract_path = tesseract_path.strip()
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Sử dụng Tesseract tại: {tesseract_path}")
            else:
                # Kiểm tra môi trường để quyết định đường dẫn tesseract mặc định
                if os.name == 'nt':  # Windows
                    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                else:  # Linux/Unix
                    tesseract_path = "/usr/bin/tesseract"
                
                # Thiết lập tesseract_cmd
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Sử dụng Tesseract mặc định tại: {tesseract_path}")
        
        # Cấu hình cho OCR - lấy trực tiếp từ biến môi trường
        ocr_language = os.getenv("OCR_LANGUAGE", "vie")
        dpi = int(os.getenv("OCR_DPI", "600"))  # Mặc định 600 nếu không có giá trị
        ocr_config = os.getenv("OCR_CONFIG", "")
        logger.info(f"Cấu hình OCR: language={ocr_language}, dpi={dpi}, config={ocr_config}")
        
        # Đặt đường dẫn poppler
        poppler_path = os.getenv("POPPLER_PATH")

        # Kiểm tra nếu đang trong môi trường Docker
        docker_env = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        if docker_env:
            # Trong Docker, luôn sử dụng đường dẫn Linux cho poppler
            poppler_path = "/usr/bin"
            logger.info(f"Đang chạy trong Docker, sử dụng Poppler Linux: {poppler_path}")
        else:
            # Xử lý cho môi trường không phải Docker
            if poppler_path:
                # Đảm bảo không có dấu ngoặc kép và khoảng trắng thừa
                poppler_path = poppler_path.replace('"', '').strip()
                logger.info(f"Sử dụng Poppler từ biến môi trường: {poppler_path}")
                
                # Kiểm tra đường dẫn Poppler tồn tại
                if not os.path.exists(poppler_path):
                    logger.error(f"Đường dẫn Poppler không tồn tại: {poppler_path}")
                    # Kiểm tra môi trường để quyết định đường dẫn mặc định
                    if os.name == 'nt':  # Windows
                        default_path = r"C:\poppler\bin"
                    else:  # Linux/Unix
                        default_path = "/usr/bin"
                
                if os.path.exists(default_path):
                    logger.info(f"Sử dụng đường dẫn Poppler mặc định: {default_path}")
                    poppler_path = default_path
                else:
                    raise FileNotFoundError(f"Không tìm thấy Poppler tại: {default_path}")
            else:
                # Kiểm tra môi trường để quyết định đường dẫn mặc định
                if os.name == 'nt':  # Windows
                    poppler_path = r"C:\poppler\bin"
                else:  # Linux/Unix
                    poppler_path = "/usr/bin"
                
                if not os.path.exists(poppler_path):
                    raise FileNotFoundError(f"Poppler không được cài đặt tại: {poppler_path}")
                logger.info(f"Sử dụng Poppler từ đường dẫn mặc định: {poppler_path}")
        
        # Chuyển PDF thành ảnh
        try:
            logger.info(f"Chuyển đổi PDF sang ảnh với poppler_path={poppler_path}")
            # Sử dụng đường dẫn chắc chắn và dành thời gian lâu hơn cho quá trình chuyển đổi
            images = convert_from_path(
                pdf_path=pdf_path,
                dpi=dpi,
                poppler_path=poppler_path,
                timeout=60  # Tăng thời gian timeout
            )
            logger.info(f"Đã chuyển đổi PDF thành {len(images)} hình ảnh")
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi PDF sang ảnh: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi chuyển đổi PDF sang ảnh: {str(e)}")
        
        all_texts = []
        
        # Xử lý từng trang
        for i, image in enumerate(images):
            logger.info(f"Đang xử lý OCR cho trang {i+1}/{len(images)}")
            
            # Tiền xử lý ảnh để cải thiện chất lượng nhận dạng (tuỳ chọn)
            # Áp dụng các kỹ thuật tiền xử lý ảnh nếu cần
            
            # Sử dụng Tesseract để OCR với các cấu hình nâng cao
            text = pytesseract.image_to_string(image, lang=ocr_language, config=ocr_config)
            
            if text:
                all_texts.append(text)
                logger.info(f"Đã trích xuất {len(text)} ký tự từ trang {i+1}")
            else:
                logger.warning(f"Không tìm thấy văn bản trên trang {i+1}")
        
        if not all_texts:
            logger.error("Không thể trích xuất văn bản nào từ PDF sử dụng Tesseract")
            return ""
        
        combined_text = "\n\n".join(all_texts)
        logger.info(f"OCR hoàn tất, đã trích xuất {len(combined_text)} ký tự")
        return combined_text
        
    except ImportError as e:
        logger.error(f"Thiếu thư viện cho Tesseract OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cần cài đặt thư viện: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi khi sử dụng Tesseract OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi OCR với Tesseract: {str(e)}")


# Hàm đọc nội dung file từ các định dạng khác nhau
async def read_file_content(file_path: str, ocr_method: str = "tesseract") -> str:
    """
    Đọc nội dung của file từ đường dẫn
    
    Args:
        file_path: Đường dẫn tới file cần đọc
        ocr_method: Phương pháp OCR, có thể là 'tesseract', 'google_vision', hoặc 'auto'
    
    Returns:
        Nội dung văn bản từ file
    """
    logger = logging.getLogger(__name__)
    
    # Chuyển đổi đường dẫn Windows sang Docker
    if os.getenv('DOCKER_ENV', 'false').lower() == 'true' and ('C:' in file_path or '\\' in file_path):
        # Chuẩn hóa đường dẫn
        file_path = file_path.replace('\\', '/')
        
        # Kiểm tra nếu file_path chứa '/uploads/'
        if '/uploads/' in file_path:
            # Trích xuất phần sau '/uploads/'
            relative_path = file_path.split('/uploads/')[1]
            docker_file_path = f"/app/uploads/{relative_path}"
            logger.info(f"Chuyển đổi đường dẫn từ Windows: {file_path} -> Docker: {docker_file_path}")
            file_path = docker_file_path
        else:
            # Thử trích xuất từ '/uploadfiles/'
            if '/uploadfiles/' in file_path:
                relative_path = file_path.split('/uploadfiles/')[1]
                docker_file_path = f"/app/uploadfiles/{relative_path}"
            else:
                # Fallback - chỉ lấy tên file
                file_name = os.path.basename(file_path)
                # Kiểm tra xem file tồn tại ở thư mục nào
                uploads_path = f"/app/uploads/{file_name}"
                uploadfiles_path = f"/app/uploadfiles/{file_name}"
                
                if os.path.exists(uploads_path):
                    docker_file_path = uploads_path
                else:
                    docker_file_path = uploadfiles_path
            
            logger.info(f"Chuyển đổi đường dẫn từ Windows: {file_path} -> Docker: {docker_file_path}")
            file_path = docker_file_path
    
    content = ""
    file_extension = file_path.split(".")[-1].lower()
    
    try:
        logger.info(f"Đang đọc file {file_path} với định dạng {file_extension}, ocr_method={ocr_method}")
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            logger.error(f"File không tồn tại tại đường dẫn: {file_path}")
            # Thử tìm trong thư mục khác nếu chưa tìm thấy
            alt_file_name = os.path.basename(file_path)
            alt_paths = [
                f"/app/uploads/{alt_file_name}",
                f"/app/uploadfiles/{alt_file_name}"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path) and alt_path != file_path:
                    logger.info(f"Tìm thấy file tại đường dẫn thay thế: {alt_path}")
                    file_path = alt_path
                    break
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Không tìm thấy file tại: {file_path} hoặc các đường dẫn thay thế")
        
        if file_extension == "txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Thử với encoding khác nếu utf-8 không hoạt động
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
                    
        elif file_extension == "pdf":
            try:
                # Thử sử dụng pdfplumber trước
                logger.info("Đang thử đọc PDF bằng pdfplumber")
                with pdfplumber.open(file_path) as pdf:
                    # Kiểm tra xem có pages không
                    if len(pdf.pages) == 0:
                        raise ValueError("PDF không có trang nào")
                        
                    # Thử trích xuất text từ từng trang
                    page_texts = []
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            page_texts.append(page_text)
                        else:
                            logger.warning(f"Không thể trích xuất văn bản từ trang {i+1}")
                    
                    # Nếu không trích xuất được text gì, chuyển sang OCR
                    if not page_texts:
                        logger.info("Không thể trích xuất text trực tiếp, chuyển sang OCR")
                        if ocr_method == "tesseract" or ocr_method == "auto":
                            content = await ocr_with_tesseract(file_path)
                            if not content and ocr_method == "auto":
                                logger.info("Tesseract OCR không thành công, thử Google Vision")
                                content = await ocr_with_google_vision(file_path)
                        else:  # google_vision
                            content = await ocr_with_google_vision(file_path)
                            
                        if not content:
                            raise ValueError("OCR không thể trích xuất văn bản từ PDF")
                        return content
                    
                    content = "\n".join(page_texts)
                    
                # Nếu văn bản trích xuất quá ngắn, có thể là PDF dạng scan
                if len(content) < 100:
                    logger.warning("Văn bản trích xuất quá ngắn, có thể là PDF dạng scan. Chuyển sang OCR...")
                    if ocr_method == "tesseract" or ocr_method == "auto":
                        content = await ocr_with_tesseract(file_path)
                        if not content and ocr_method == "auto":
                            logger.info("Tesseract OCR không thành công, thử Google Vision")
                            content = await ocr_with_google_vision(file_path)
                    else:  # google_vision
                        content = await ocr_with_google_vision(file_path)
                    
                    if not content:
                        raise ValueError("OCR không thể trích xuất văn bản từ PDF")
                    
            except Exception as e:
                if "OCR không thể trích xuất văn bản từ PDF" in str(e):
                    raise
                
                logger.error(f"Lỗi khi đọc PDF với phương thức thông thường: {str(e)}")
                logger.info("Thử phương pháp OCR sau lỗi")
                
                # Thử OCR nếu phương thức thông thường thất bại
                try:
                    if ocr_method == "tesseract" or ocr_method == "auto":
                        content = await ocr_with_tesseract(file_path)
                        if not content and ocr_method == "auto":
                            logger.info("Tesseract OCR không thành công, thử Google Vision")
                            content = await ocr_with_google_vision(file_path)
                    else:  # google_vision
                        content = await ocr_with_google_vision(file_path)
                        
                    if not content:
                        raise ValueError("OCR không thể trích xuất văn bản từ PDF")
                except Exception as ocr_err:
                    logger.error(f"Lỗi khi thực hiện OCR: {str(ocr_err)}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Không thể đọc nội dung PDF. Lỗi: {str(e)}. OCR cũng thất bại: {str(ocr_err)}"
                    )
                
        elif file_extension == "docx":
            try:
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
                if not content.strip():
                    logger.warning("DOCX không có nội dung văn bản")
            except Exception as e:
                logger.error(f"Lỗi khi đọc DOCX: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Không thể đọc nội dung DOCX. Lỗi: {str(e)}")
                
        elif file_extension in ["csv", "xlsx"]:
            try:
                if file_extension == "csv":
                    try:
                        df = pd.read_csv(file_path, encoding="utf-8")
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding="latin-1")
                else:
                    df = pd.read_excel(file_path)
                content = df.to_string()
            except Exception as e:
                logger.error(f"Lỗi khi đọc {file_extension}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Không thể đọc nội dung {file_extension}. Lỗi: {str(e)}")
                
        else:
            raise HTTPException(status_code=400, detail=f"Định dạng file {file_extension} không được hỗ trợ!")
        
        # Kiểm tra nếu không có nội dung
        if not content.strip():
            logger.warning(f"File {file_path} không có nội dung văn bản")
            raise HTTPException(status_code=400, detail="File không có nội dung văn bản để phân đoạn.")
            
        logger.info(f"Đã đọc thành công file {file_path}, kích thước nội dung: {len(content)} ký tự")
        return content
        
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file {file_path}")
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file {file_path}")
    except Exception as e:
        logger.error(f"Lỗi không xác định khi đọc file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file: {str(e)}")


# Hàm xử lý chunking văn bản
async def process_text(text: str, config: Optional[ChunkingConfig] = None):
    """Xử lý chunking bằng phương pháp Semantic Chunker"""
    logger = logging.getLogger(__name__)
    
    if not config:
        config = ChunkingConfig(verbose_logging=True)
    else:
        config_dict = config.dict()
        config_dict["verbose_logging"] = True
        config = ChunkingConfig(**config_dict)
        
    if config.language == "vietnamese":
        if UNDERTHESEA_AVAILABLE:
            logger.info("Underthesea được kích hoạt cho xử lý tiếng Việt")
        else:
            logger.warning("Underthesea không khả dụng, sử dụng NLTK cho tiếng Việt")
            
    if config.use_clustering:
        logger.info(f"=== Thực hiện semantic chunking với phương pháp {config.clustering_method.upper()} ===")
        logger.info(f"Cấu hình clustering: threshold={config.threshold}, preserve_order={config.preserve_order}, verbose_logging={config.verbose_logging}")
    else:
        logger.info(f"=== Thực hiện semantic chunking với phương pháp TRUYỀN THỐNG ===")
        logger.info(f"Cấu hình: verbose_logging={config.verbose_logging}")
    
    chunker = SemanticChunker(config)
    chunks = await chunker.split_text(text)
    
    logger.info(f"Kết quả chunking: {len(chunks)} chunks")
    for i, chunk_data in enumerate(chunks[:3]): 
        chunk_text = chunk_data.get("text", "") if isinstance(chunk_data, dict) else chunk_data
        logger.info(f"Chunk {i+1} (độ dài: {len(chunk_text)}): {chunk_text[:100]}...")
    
    if len(chunks) > 3:
        logger.info(f"... và {len(chunks) - 3} chunks khác")
    
    return chunks


# Hàm lưu các chunk vào cơ sở dữ liệu PostgreSQL
async def save_to_db_chunking(file_name, chunks: List[Dict[str, Any]], document_id, config=None):
    """Lưu chunking vào PostgreSQL với trường embedding là NULL và các trường mặc định."""
    logger = logging.getLogger(__name__)
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            # Sử dụng phiên bản sync đã được import
            register_vector_sync(cursor)

            config_id = None
            if config:
                cursor.execute("""
                    INSERT INTO chunking_configs (document_id, config, created_at)
                    VALUES (%s, %s, NOW())
                    RETURNING id;
                """, (document_id, json.dumps(config.dict())))
                config_id = cursor.fetchone()[0]

            for chunk_data in chunks:
                chunk_text = chunk_data.get('text')
                chunk_keywords = chunk_data.get('keywords') # Get keywords
                
                if chunk_text is not None:
                    cursor.execute("""
                        INSERT INTO chunks (document_id, chunk_text, keywords, embedding, config_id)
                        VALUES (%s, %s, %s, NULL, %s);
                    """, (document_id, chunk_text, chunk_keywords, config_id)) # Add chunk_keywords to insert

            cursor.execute("""
                UPDATE documents
                SET "isChunked" = true
                WHERE id = %s;
            """, (document_id,))

            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Lỗi khi lưu chunks vào DB: {str(e)}", exc_info=True)
        raise
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


# Hàm lấy tài liệu từ cơ sở dữ liệu
def get_document_by_id(document_id: str):
    """Lấy thông tin tài liệu từ cơ sở dữ liệu bằng document_id"""
    conn, pool = None, None
    document = None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, document_link FROM documents WHERE id = %s;
            """, (document_id,))
            document_row = cursor.fetchone()
            if document_row:
                document = {"id": document_row[0], "document_link": document_row[1]}
    except Exception as e:
        logger.error(f"Lỗi khi lấy document by id: {str(e)}")
        # Có thể raise exception hoặc trả về None tùy vào logic mong muốn
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)
    return document


# API để kiểm tra trạng thái embedding của các chunk
@router.get("/check_embedding_status/")
async def check_embedding_status(document_id: str):
    try:
        # Lấy danh sách chunk của tài liệu
        chunks = get_chunks_from_db(document_id=document_id)

        # Kiểm tra xem tất cả chunk đã được embedding chưa
        all_embedded = all(chunk[2] is not None for chunk in chunks)

        # Thống kê số lượng chunk đã embedding
        embedded_count = sum(1 for chunk in chunks if chunk[2] is not None)
        total_count = len(chunks)

        return {
            "document_id": document_id,
            "all_embedded": all_embedded,
            "embedded_count": embedded_count,
            "total_count": total_count,
            "progress_percentage": (embedded_count / total_count * 100) if total_count > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Định nghĩa hàm get_chunks_from_db
def get_chunks_from_db(document_id: str):
    conn, pool = None, None
    chunks = []
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, chunk_text, embedding FROM chunks WHERE document_id = %s;
            """, (document_id,))
            chunks = cursor.fetchall()
    except Exception as e:
        logger.error(f"Lỗi khi lấy chunks từ DB: {str(e)}")
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)
    return chunks


@router.get("/ocr/check_tesseract/")
async def check_tesseract():
    """Kiểm tra Tesseract OCR đã được cài đặt đúng hay chưa"""
    try:
        # Lấy đường dẫn Tesseract từ biến môi trường hoặc sử dụng giá trị mặc định
        tesseract_path = os.getenv("TESSERACT_CMD_PATH", "tesseract")
        
        # Thiết lập đường dẫn nếu được cung cấp
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Kiểm tra phiên bản Tesseract
        try:
            result = subprocess.run([tesseract_path, "--version"], 
                                   capture_output=True, text=True, check=True)
            version_info = result.stdout.strip()
            
            # Kiểm tra supported languages
            langs_result = subprocess.run([tesseract_path, "--list-langs"],
                                         capture_output=True, text=True)
            supported_langs = langs_result.stdout.strip().split('\n')
            # Bỏ dòng đầu tiên (thường là tiêu đề)
            if len(supported_langs) > 1:
                supported_langs = supported_langs[1:]
            
            return {
                "status": "success",
                "tesseract_path": tesseract_path,
                "version_info": version_info,
                "supported_languages": supported_langs,
                "poppler_path": os.getenv("POPPLER_PATH", "Not set"),
                "ocr_language": os.getenv("OCR_LANGUAGE", "vie"),
                "ocr_dpi": os.getenv("OCR_DPI", "300")
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Tesseract được cài đặt nhưng có lỗi khi chạy: {str(e)}",
                "tesseract_path": tesseract_path
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi kiểm tra Tesseract OCR: {str(e)}"
        )


# Hàm để lấy cấu hình OCR hiện tại
@router.get("/ocr/config/")
async def get_ocr_config():
    """Lấy cấu hình OCR hiện tại từ biến môi trường"""
    try:
        tesseract_path = os.getenv("TESSERACT_CMD_PATH", "tesseract")
        poppler_path = os.getenv("POPPLER_PATH", "")
        ocr_language = os.getenv("OCR_LANGUAGE", "eng")
        dpi = os.getenv("OCR_DPI")
        ocr_config = os.getenv("OCR_CONFIG", "")
        
        return {
            "status": "success",
            "config": {
                "tesseract_path": tesseract_path,
                "poppler_path": poppler_path,
                "ocr_language": ocr_language,
                "dpi": dpi,
                "ocr_config": ocr_config
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get OCR config: {str(e)}")


# Cập nhật schema config với các tham số mới
@router.post("/config")
async def update_config(
    threshold: Optional[float] = None,
    embedding_type: Optional[str] = None,
    model: Optional[str] = None,
    min_sentence_length: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    overlap_size: Optional[int] = None,
    clean_text: Optional[bool] = None,
    language: Optional[str] = None,
    use_clustering: Optional[bool] = None,  # Thêm tham số clustering
    preserve_order: Optional[bool] = None   # Thêm tham số bảo toàn thứ tự
):
    """Cập nhật cấu hình chunking"""
    logger = logging.getLogger(__name__)
    
    if use_clustering is not None:
        if use_clustering:
            logger.info("KÍCH HOẠT phương pháp phân cụm (clustering) trong cấu hình")
        else:
            logger.info("TẮT phương pháp phân cụm (clustering) trong cấu hình")
    
    global semantic_config
    
    if threshold is not None:
        semantic_config.threshold = threshold
    if embedding_type is not None:
        semantic_config.embedding_type = embedding_type
    if model is not None:
        semantic_config.model = model
    if min_sentence_length is not None:
        semantic_config.min_sentence_length = min_sentence_length
    if min_chunk_size is not None:
        semantic_config.min_chunk_size = min_chunk_size
    if max_chunk_size is not None:
        semantic_config.max_chunk_size = max_chunk_size
    if overlap_size is not None:
        semantic_config.overlap_size = overlap_size
    if clean_text is not None:
        semantic_config.clean_text = clean_text
    if language is not None:
        semantic_config.language = language
    if use_clustering is not None:
        semantic_config.use_clustering = use_clustering
    if preserve_order is not None:
        semantic_config.preserve_order = preserve_order
    
    # Tạo lại chunker với cấu hình mới
    chunker = SemanticChunker(config=semantic_config)
    
    # Trả về config hiện tại
    return {
        "status": "success",
        "config": {
            "threshold": semantic_config.threshold,
            "embedding_type": semantic_config.embedding_type,
            "model": semantic_config.model,
            "min_sentence_length": semantic_config.min_sentence_length,
            "min_chunk_size": semantic_config.min_chunk_size,
            "max_chunk_size": semantic_config.max_chunk_size,
            "overlap_size": semantic_config.overlap_size,
            "clean_text": semantic_config.clean_text,
            "language": semantic_config.language,
            "use_clustering": semantic_config.use_clustering,
            "preserve_order": semantic_config.preserve_order
        }
    }

# Get config endpoint
@router.get("/config")
async def get_config():
    global semantic_config
    
    if semantic_config is None:
        semantic_config = ChunkingConfig.get_default_config()
        
    return {
        "status": "success",
        "config": {
            "threshold": semantic_config.threshold,
            "embedding_type": semantic_config.embedding_type,
            "model": semantic_config.model,
            "min_sentence_length": semantic_config.min_sentence_length,
            "min_chunk_size": semantic_config.min_chunk_size,
            "max_chunk_size": semantic_config.max_chunk_size,
            "overlap_size": semantic_config.overlap_size,
            "clean_text": semantic_config.clean_text,
            "language": semantic_config.language,
            "use_clustering": semantic_config.use_clustering,
            "preserve_order": semantic_config.preserve_order
        },
        "available_models": ChunkingConfig.get_available_models(),
        "available_embedding_types": ChunkingConfig.get_available_embedding_types(),
        "available_languages": ChunkingConfig.get_available_languages()
    }
