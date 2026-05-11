import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from database.db_connection import get_pg_connection, return_pg_connection
import numpy as np
import uuid
import os
from llms.onlinellms import OnlineLLMs
from llms.localLllms import LocalLlms
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from scipy.fftpack import dct, idct
from sklearn.decomposition import PCA

# Load env vars từ file .env
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter()  # Tạo router cho chunking


# Model Pydantic cho API key
class ApiKeys(BaseModel):
    openai_api_key: Optional[str] = Field(None, description="API key cho OpenAI (GPT)")
    gemini_api_key: Optional[str] = Field(None, description="API key cho Google Gemini")
    cohere_api_key: Optional[str] = Field(None, description="API key cho Cohere")


# Model Pydantic cho cấu hình embedding
class EmbeddingConfig(BaseModel):
    model_type: str = "online"  # "online" hoặc "local"
    model_name: str = "text-embedding-3-small"  # Tên mô hình mặc định
    provider: Optional[str] = "openai"  # Nhà cung cấp: "openai", "gemini", "cohere", "local"
    batch_size: int = 10  # Số lượng chunk xử lý mỗi lần
    update_document_status: bool = True  # Cập nhật trạng thái tài liệu sau khi hoàn thành
    api_keys: Optional[ApiKeys] = None  # API keys cho các dịch vụ
    endpoint: Optional[str] = None  # URL endpoint cho mô hình local
    target_dimension: int = 1536  # Kích thước embedding mong muốn (quay lại 1536)
    embedding_method: str = "dct"  # Phương pháp điều chỉnh kích thước: linear, redistribute, dct, pca, pad
    # Thêm các trường mới để tương thích với LlmConfig
    embedding_type: Optional[str] = None  # "online" hoặc "local" - tương thích với LlmConfig
    embedding_provider: Optional[str] = None  # Provider cụ thể cho embedding


# Biến toàn cục lưu trữ cấu hình embedding
embedding_config = EmbeddingConfig(
    api_keys=ApiKeys(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
)
embedding_model = None  # Sẽ được khởi tạo sau khi cấu hình


# Lưu API keys vào biến môi trường
def update_api_keys():
    if embedding_config.api_keys:
        if embedding_config.api_keys.openai_api_key:
            os.environ["OPENAI_API_KEY"] = embedding_config.api_keys.openai_api_key
        if embedding_config.api_keys.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = embedding_config.api_keys.gemini_api_key
        if embedding_config.api_keys.cohere_api_key:
            os.environ["COHERE_API_KEY"] = embedding_config.api_keys.cohere_api_key


# Hàm để khởi tạo mô hình dựa trên cấu hình
def initialize_embedding_model():
    global embedding_model

    # Cập nhật API keys trước khi khởi tạo model
    update_api_keys()
    
    # Chuẩn bị cấu hình với api_key trực tiếp
    config_dict = embedding_config.dict()
    api_key = None 
    
    # Ưu tiên sử dụng provider trực tiếp từ config
    provider = embedding_config.provider.lower() if embedding_config.provider else None
    
    # Trích xuất API key từ cấu trúc nested api_keys theo provider
    if embedding_config.api_keys:
        if provider == "gemini":
            api_key = embedding_config.api_keys.gemini_api_key or os.getenv("GEMINI_API_KEY")
            print(f"Using Gemini API key based on provider field: {embedding_config.model_name}")
        elif provider == "cohere":
            api_key = embedding_config.api_keys.cohere_api_key or os.getenv("COHERE_API_KEY")
            print(f"Using Cohere API key based on provider field: {embedding_config.model_name}")
        elif provider == "openai":
            api_key = embedding_config.api_keys.openai_api_key or os.getenv("OPENAI_API_KEY")
            print(f"Using OpenAI API key based on provider field: {embedding_config.model_name}")
        else:
            # Nếu không có provider rõ ràng, sử dụng heuristic từ tên model
            model_name_lower = embedding_config.model_name.lower()
            
            if "gemini" in model_name_lower:
                api_key = embedding_config.api_keys.gemini_api_key or os.getenv("GEMINI_API_KEY")
                print(f"Using Gemini API key based on model name: {embedding_config.model_name}")
            elif "cohere" in model_name_lower or "command" in model_name_lower:
                api_key = embedding_config.api_keys.cohere_api_key or os.getenv("COHERE_API_KEY")
                print(f"Using Cohere API key based on model name: {embedding_config.model_name}")
            elif any(name in model_name_lower for name in ["gpt", "text-davinci", "davinci", "text-embedding"]):
                api_key = embedding_config.api_keys.openai_api_key or os.getenv("OPENAI_API_KEY")
                print(f"Using OpenAI API key based on model name: {embedding_config.model_name}")
            else:
                # Default to OpenAI
                api_key = embedding_config.api_keys.openai_api_key or os.getenv("OPENAI_API_KEY")
                print(f"Using default OpenAI API key: {embedding_config.model_name}")
    
    # Kiểm tra API key
    if embedding_config.model_type == "online" and not api_key:
        provider_name = provider or "nhà cung cấp không xác định"
        raise ValueError(f"API key không được cung cấp cho {provider_name}")

    # Khởi tạo model - ưu tiên embedding_type nếu có, fallback về model_type
    embedding_type = embedding_config.embedding_type or embedding_config.model_type
    
    if embedding_type == "online":
        # Khởi tạo model với bất kỳ tên model nào
        embedding_model = OnlineLLMs(
            model_name=embedding_config.model_name,
            api_key=api_key,
            provider=provider,  # Truyền thêm provider
            endpoint=embedding_config.endpoint
        )
    elif embedding_type == "local":
        # Khởi tạo model local với SentenceTransformer
        from llms.local_embedding import LocalEmbedding
        embedding_model = LocalEmbedding(model_name=embedding_config.model_name)
    else:
        raise ValueError(f"Embedding type không hợp lệ: {embedding_type}")
    return embedding_model


# Khởi tạo mô hình mặc định
try:
    embedding_model = initialize_embedding_model()
except ValueError as e:
    print(f"Lỗi khi khởi tạo mô hình mặc định: {str(e)}")
    print("Hãy cấu hình API key qua endpoint /embedding/config/")


# API để lấy cấu hình embedding hiện tại
@router.get("/embedding/config/")
async def get_embedding_config():
    """Lấy cấu hình embedding hiện tại"""
    # Tạo bản sao cấu hình để trả về
    config_copy = embedding_config.dict()

    # Che giấu API keys trong phản hồi bằng cách thay thế
    if config_copy.get("api_keys"):
        for key in config_copy["api_keys"]:
            if config_copy["api_keys"][key]:
                config_copy["api_keys"][key] = "********"  # Che giấu API key

    return config_copy


# Model cho việc cập nhật API key
class ApiKeyUpdate(BaseModel):
    key_type: str  # "openai", "gemini", "cohere"
    api_key: str


# API để cập nhật API key riêng lẻ
@router.post("/embedding/config/api-key/")
async def update_api_key(key_update: ApiKeyUpdate):
    """Cập nhật API key riêng lẻ"""
    global embedding_config

    if not embedding_config.api_keys:
        embedding_config.api_keys = ApiKeys()

    if key_update.key_type == "openai":
        embedding_config.api_keys.openai_api_key = key_update.api_key
    elif key_update.key_type == "gemini":
        embedding_config.api_keys.gemini_api_key = key_update.api_key
    elif key_update.key_type == "cohere":
        embedding_config.api_keys.cohere_api_key = key_update.api_key
    else:
        raise HTTPException(status_code=400, detail=f"Loại key không hợp lệ: {key_update.key_type}")

    # Cập nhật biến môi trường
    update_api_keys()

    return {"message": f"Đã cập nhật API key cho {key_update.key_type} thành công"}


# API để cập nhật cấu hình embedding
@router.post("/embedding/config/")
async def update_embedding_config(config: EmbeddingConfig):
    """Cập nhật cấu hình embedding"""
    global embedding_config
    global embedding_model
    
    # Ghi log cấu hình mới
    logger.info(f"Cập nhật cấu hình embedding mới: {config.dict()}")
    logger.info(f"Phương pháp điều chỉnh kích thước: {config.embedding_method}")
    
    # Lưu cấu hình mới
    embedding_config = config
    
    # Cập nhật API keys vào biến môi trường
    update_api_keys()
    
    # Khởi tạo lại model với cấu hình mới
    try:
        embedding_model = initialize_embedding_model()
        return {"status": "success", "message": "Đã cập nhật cấu hình embedding"}
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo lại model: {str(e)}")
        return {"status": "error", "message": f"Lỗi khi khởi tạo lại model: {str(e)}"}


# API để quản lý cache của local embedding
@router.get("/embedding/cache/info/")
async def get_embedding_cache_info():
    """Lấy thông tin về cache của local embedding models"""
    try:
        from llms.local_embedding import LocalEmbedding
        cache_info = LocalEmbedding.get_cache_info()
        return {
            "status": "success",
            "cached_models": cache_info,
            "total_cached": len(cache_info)
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin cache: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.post("/embedding/cache/clear/")
async def clear_embedding_cache():
    """Xóa tất cả model khỏi cache"""
    try:
        from llms.local_embedding import LocalEmbedding
        LocalEmbedding.clear_cache()
        return {
            "status": "success",
            "message": "Đã xóa tất cả model khỏi cache"
        }
    except Exception as e:
        logger.error(f"Lỗi khi xóa cache: {str(e)}")
        return {"status": "error", "message": str(e)}


# Dependency để đảm bảo mô hình đã được khởi tạo
def get_embedding_model():
    if embedding_model is None:
        try:
            return initialize_embedding_model()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return embedding_model


# Kiểm tra API key có phù hợp với model được chọn
def validate_api_key_for_model():
    # Trích xuất API key từ cấu trúc nested api_keys
    api_key = None
    
    if embedding_config.model_type == "online":
        # Ưu tiên sử dụng provider trực tiếp từ config
        provider = embedding_config.provider.lower() if embedding_config.provider else None
        
        if provider == "gemini":
            api_key = embedding_config.api_keys.gemini_api_key if embedding_config.api_keys else None
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="GEMINI_API_KEY chưa được cấu hình cho mô hình Gemini")
        elif provider == "cohere":
            api_key = embedding_config.api_keys.cohere_api_key if embedding_config.api_keys else None
            api_key = api_key or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="COHERE_API_KEY chưa được cấu hình cho mô hình Cohere")
        elif provider == "openai":
            api_key = embedding_config.api_keys.openai_api_key if embedding_config.api_keys else None
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="OPENAI_API_KEY chưa được cấu hình cho mô hình OpenAI")
        else:
            # Nếu không có provider rõ ràng, sử dụng heuristic từ tên model
            model_name_lower = embedding_config.model_name.lower()
            
            if "gemini" in model_name_lower:
                api_key = embedding_config.api_keys.gemini_api_key if embedding_config.api_keys else None
                api_key = api_key or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=400, detail="GEMINI_API_KEY chưa được cấu hình cho mô hình Gemini")
            elif "cohere" in model_name_lower or "command" in model_name_lower:
                api_key = embedding_config.api_keys.cohere_api_key if embedding_config.api_keys else None
                api_key = api_key or os.getenv("COHERE_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=400, detail="COHERE_API_KEY chưa được cấu hình cho mô hình Cohere")
            elif any(name in model_name_lower for name in ["gpt", "text-davinci", "davinci", "text-embedding"]):
                api_key = embedding_config.api_keys.openai_api_key if embedding_config.api_keys else None
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=400, detail="OPENAI_API_KEY chưa được cấu hình cho mô hình OpenAI")
            else:
                # Kiểm tra key mặc định cho model khác
                api_key = embedding_config.api_keys.openai_api_key if embedding_config.api_keys else None
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=400, detail=f"API key chưa được cấu hình cho mô hình: {embedding_config.model_name}")
    
    return api_key


# API để tạo embedding cho các chunk đã được upload trong csdl
@router.post("/embedding/")
async def create_embeddings_for_document(
        document_id: str = Query(..., description="UUID của tài liệu"),
        chunk_ids: Optional[List[str]] = Query(None, description="Danh sách các chunk_id cụ thể"),
        model: Any = Depends(get_embedding_model)
):
    """
    API tạo embeddings cho các chunk đã được lưu trong cơ sở dữ liệu.
    - Nếu document_id (UUID) được cung cấp, tạo embeddings cho tất cả chunks của tài liệu có trường embedding là NULL.
    - Nếu document_id và chunk_ids được cung cấp, chỉ tạo embeddings cho các chunk có ID trong danh sách chunk_ids và có trường embedding là NULL.
    """
    try:
        # Kiểm tra API key trước khi thực hiện
        api_key = validate_api_key_for_model()
        print(f"API key validated for embedding: {'Yes' if api_key else 'No'}")
        print(f"Using provider: {embedding_config.provider or 'Not explicitly set'}")

        # Kiểm tra document_id có hợp lệ không
        try:
            document_uuid = uuid.UUID(document_id)  # Chuyển chuỗi thành UUID
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document_id format.")

        # Lấy chunks từ cơ sở dữ liệu
        chunks = get_chunks_from_db(document_id=str(document_uuid), chunk_ids=chunk_ids)

        # Kiểm tra nếu không có chunks nào
        if not chunks:
            raise HTTPException(status_code=404, detail="No chunks found for the given document or chunk ids.")

        # Thực hiện embedding theo batch_size được cấu hình
        all_embeddings = []
        batch_size = embedding_config.batch_size

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            # Thực hiện embedding bằng mô hình được cấu hình
            embedding_type = embedding_config.embedding_type or embedding_config.model_type
            
            if embedding_type == "online":
                batch_embeddings = [model.generate_embedding(chunk[1]) for chunk in batch_chunks]
            else:  # embedding_type == "local"
                # Xử lý với mô hình local SentenceTransformer
                batch_texts = [chunk[1] for chunk in batch_chunks]
                batch_embeddings = [model.generate_embedding(text) for text in batch_texts]

            all_embeddings.extend(batch_embeddings)

        # Cập nhật embedding vào cơ sở dữ liệu
        update_embeddings_in_db(document_uuid, chunks, all_embeddings)

        # Kiểm tra và cập nhật trạng thái embedding của tài liệu nếu được cấu hình
        if embedding_config.update_document_status and check_all_chunks_embedded(document_id):
            # Nếu tất cả các chunk đều đã có embedding, cập nhật trạng thái isEmbeddingDone
            update_embedding_status(document_id)

        return {
            "document_id": document_id,
            "chunk_ids": chunk_ids if chunk_ids else "N/A",
            "chunks_processed": len(chunks),
            "batch_size": batch_size,
            "model_type": embedding_config.model_type,
            "model_name": embedding_config.model_name,
            "provider": embedding_config.provider,
            "status": "Completed"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Hàm kiểm tra xem tất cả các chunk của tài liệu đã có embedding chưa
def check_all_chunks_embedded(document_id: str, chunk_ids: List[str] = None) -> bool:
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            # Nếu có chunk_ids, kiểm tra từng chunk cụ thể
            if chunk_ids:
                chunk_ids = [str(uuid.UUID(chunk_id)) for chunk_id in chunk_ids]
                cursor.execute("""
                    SELECT id, embedding FROM chunks 
                    WHERE document_id = %s AND id = ANY(%s::uuid[]);
                """, (document_id, chunk_ids))
            else:
                # Nếu không có chunk_ids, kiểm tra tất cả các chunk của tài liệu
                cursor.execute("""
                    SELECT id, embedding FROM chunks WHERE document_id = %s;
                """, (document_id,))

            chunks = cursor.fetchall()
            conn.commit()
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra embedding status: {e}")
        return False # Giả định là chưa embedded nếu có lỗi
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)
    # Kiểm tra nếu có bất kỳ chunk nào có embedding là NULL
    for chunk in chunks:
        chunk_id, embedding = chunk
        if embedding is None:  # Nếu embedding là NULL, trả về False
            return False

    # Nếu không có chunk nào có embedding NULL, trả về True
    return True


# Hàm lấy các chunk từ cơ sở dữ liệu
def get_chunks_from_db(document_id: str = None, chunk_ids: List[str] = None):
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            if document_id:
                if chunk_ids:
                    # Cập nhật câu lệnh SQL để ép kiểu chunk_ids thành UUID[] khi truy vấn
                    cursor.execute("""
                        SELECT id, chunk_text FROM chunks WHERE document_id = %s AND id = ANY(%s::uuid[]) AND embedding IS NULL;
                    """, (document_id, chunk_ids))
                else:
                    cursor.execute("""
                        SELECT id, chunk_text FROM chunks WHERE document_id = %s AND embedding IS NULL;
                    """, (document_id,))
            elif chunk_ids:
                # Đảm bảo ép kiểu chunk_ids thành UUID[] khi không có document_id
                cursor.execute("""
                    SELECT id, chunk_text FROM chunks WHERE id = ANY(%s::uuid[]) AND embedding IS NULL;
                """, (chunk_ids,))
            else:
                raise HTTPException(status_code=400, detail="Either document_id or chunk_ids must be provided.")

            chunks = cursor.fetchall()
            conn.commit()
    except Exception as e:
        logger.error(f"Lỗi khi lấy chunks từ DB: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy chunks từ DB: {e}")
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)
    return chunks


def pad_embedding_vector(vector: List[float], target_dim: int = 1536, method: str = "dct") -> List[float]:
    """
    Điều chỉnh kích thước vector embedding thành target_dim bằng nhiều phương pháp
    
    Args:
        vector: Vector embedding gốc
        target_dim: Kích thước đích (mặc định 1536)
        method: Phương pháp điều chỉnh kích thước
            - "linear": Biến đổi tuyến tính cải tiến (mặc định)
            - "redistribute": Phân phối lại giá trị
            - "dct": Discrete Cosine Transform
            - "pca": PCA (dùng cho giảm chiều)
            - "pad": Padding đơn thuần (thêm 0 - phương pháp cũ)
    
    Returns:
        Vector đã được điều chỉnh kích thước
    """
    current_dim = len(vector)
    
    # Trường hợp vector đã đúng kích thước
    if current_dim == target_dim:
        return vector
    
    # Trường hợp giảm chiều từ cao xuống target_dim
    if current_dim > target_dim:
        if method == "pca":
            # Sử dụng PCA để giảm chiều
            try:
                pca = PCA(n_components=target_dim)
                # Reshape vector để phù hợp với PCA
                vector_2d = np.array(vector).reshape(1, -1)
                # Áp dụng PCA
                reduced = pca.fit_transform(vector_2d)
                return reduced[0].tolist()
            except Exception as e:
                print(f"Lỗi khi áp dụng PCA: {str(e)}")
                # Fallback: Cắt bớt nếu PCA không thành công
                return vector[:target_dim]
        else:
            # Cắt bớt nếu vector quá dài hoặc không dùng PCA
            return vector[:target_dim]
    
    # Trường hợp tăng chiều từ thấp lên target_dim
    if method == "linear":
        # Phương pháp 1: Biến đổi tuyến tính cải tiến
        # Sử dụng seed cố định để đảm bảo nhất quán
        np.random.seed(42)
        
        # Tạo ma trận biến đổi tĩnh bằng phép chiếu trực giao
        transform_matrix = np.random.normal(0, 1/np.sqrt(current_dim), (current_dim, target_dim))
        
        # Chuẩn hóa trực giao cho ma trận
        for i in range(min(current_dim, target_dim)):
            # Giữ nguyên current_dim chiều đầu tiên (identity mapping)
            if i < current_dim:
                transform_matrix[i, i] = 1.0
        
        # Áp dụng biến đổi
        transformed = np.dot(np.array(vector), transform_matrix)
        
        # Chuẩn hóa độ lớn để giữ nguyên tỷ lệ
        norm_ratio = np.linalg.norm(vector) / np.linalg.norm(transformed)
        transformed = transformed * norm_ratio
        
        return transformed.tolist()
    
    elif method == "redistribute":
        # Phương pháp 2: Phân phối lại giá trị
        result = np.zeros(target_dim)
        
        # Sao chép các giá trị gốc vào đầu vector
        result[:current_dim] = vector
        
        # Phân phối thông tin từ vector gốc vào phần còn lại
        remaining_dims = target_dim - current_dim
        
        # Tạo pattern lặp lại có trọng số
        for i in range(remaining_dims):
            # Lấy vị trí trong vector gốc (lặp lại nếu cần)
            source_idx = i % current_dim
            # Tạo trọng số giảm dần theo khoảng cách
            weight = 0.5 / (1 + i // current_dim)
            # Gán giá trị có trọng số
            result[current_dim + i] = vector[source_idx] * weight
        
        # Đảm bảo tiêu chuẩn hóa phù hợp
        orig_norm = np.linalg.norm(vector)
        result_norm = np.linalg.norm(result)
        if result_norm > 0:  # Tránh chia cho 0
            result = result * (orig_norm / result_norm)
        
        return result.tolist()
    
    elif method == "dct":
        # Phương pháp 3: Discrete Cosine Transform
        try:
            # Chuyển đổi sang miền tần số
            dct_coeffs = dct(vector, type=2, norm='ortho')
            
            # Tạo hệ số tần số cao bằng 0
            padded_coeffs = np.zeros(target_dim)
            padded_coeffs[:current_dim] = dct_coeffs
            
            # Chuyển ngược về miền không gian
            upsampled = idct(padded_coeffs, type=2, norm='ortho')
            
            # Chuẩn hóa để đảm bảo độ lớn tương tự
            norm_ratio = np.linalg.norm(vector) / np.linalg.norm(upsampled)
            upsampled = upsampled * norm_ratio
            
            return upsampled.tolist()
        except Exception as e:
            print(f"Lỗi khi áp dụng DCT: {str(e)}")
            # Fallback: Sử dụng phương pháp padding nếu DCT không khả dụng
            padding = [0.0] * (target_dim - current_dim)
            return vector + padding
    
    else:  # method == "pad" hoặc bất kỳ giá trị khác
        # Phương pháp cũ: Padding đơn thuần với 0
        padding = [0.0] * (target_dim - current_dim)
        return vector + padding


# Hàm cập nhật embeddings vào cơ sở dữ liệu
def update_embeddings_in_db(document_id, chunks, embeddings):
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            for idx, (chunk_id, chunk_text) in enumerate(chunks):
                # Áp dụng điều chỉnh kích thước cho vector embedding
                padded_embedding = pad_embedding_vector(
                    embeddings[idx], 
                    target_dim=embedding_config.target_dimension,
                    method=embedding_config.embedding_method
                )
                cursor.execute("""
                    UPDATE chunks
                    SET embedding = %s, isEmbeddingDone = true
                    WHERE document_id = %s AND id = %s AND embedding IS NULL;
                """, (np.array(padded_embedding).tolist(), str(document_id), chunk_id))
            conn.commit()
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"Lỗi khi cập nhật embeddings vào DB: {e}")
        raise
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


# Hàm cập nhật trạng thái isEmbeddingDone trong cơ sở dữ liệu
def update_embedding_status(document_id: str):
    conn, pool = None, None
    try:
        conn, pool = get_pg_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE documents
                SET "isEmbeddingDone" = true
                WHERE "id" = %s;
            """, (document_id,))
            conn.commit()
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating embedding status: {str(e)}")
    finally:
        if conn and pool:
            return_pg_connection(conn, pool)


# API để chỉ cập nhật phương pháp điều chỉnh kích thước
@router.post("/embedding/dimension/")
async def update_embedding_dimension_method(
    target_dimension: int = 1536,
    method: str = "linear"
):
    """Cập nhật phương pháp điều chỉnh kích thước vector embedding"""
    global embedding_config
    
    # Kiểm tra method hợp lệ
    valid_methods = ["linear", "redistribute", "dct", "pca", "pad"]
    if method not in valid_methods:
        return {
            "status": "error",
            "message": f"Phương pháp không hợp lệ. Các phương pháp hỗ trợ: {', '.join(valid_methods)}"
        }
    
    # Cập nhật cấu hình
    embedding_config.target_dimension = target_dimension
    embedding_config.embedding_method = method
    
    logger.info(f"Đã cập nhật phương pháp điều chỉnh kích thước: {method}, kích thước đích: {target_dimension}")
    
    return {
        "status": "success",
        "message": f"Đã cập nhật phương pháp '{method}' và kích thước đích {target_dimension}",
        "config": {
            "target_dimension": embedding_config.target_dimension,
            "embedding_method": embedding_config.embedding_method
        }
    }