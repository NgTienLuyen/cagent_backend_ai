# services/keyword_extractor.py
import logging
from typing import List, Dict, Any, Optional

try:
    from underthesea import word_tokenize, pos_tag
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    logging.warning("Thư viện underthesea không được tìm thấy. NLP keyword extraction sẽ bị hạn chế.")

logger = logging.getLogger(__name__)

# Danh sách stop words tiếng Việt cơ bản - có thể mở rộng thêm
VIETNAMESE_STOP_WORDS = [
    "và", "là", "của", "thì", "mà", "rằng", "rồi", "đã", "đang", "sẽ", "khi", "để", "cho", "từ", "cũng", "như",
    "các", "có", "không", "được", "trong", "ra", "vào", "lên", "xuống", "trên", "dưới", "trước", "sau", "đây",
    "đó", "ấy", "này", "kia", "khác", "những", "nhiều", "ít", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy",
    "tám", "chín", "mười", "ông", "bà", "anh", "chị", "em", "tôi", "chúng ta", "chúng tôi", "bạn", "họ",
    "mình", "mọi", "mỗi", "điều", "việc", "thứ", "lần", "theo", "tại", "đến", "vì", "nên", "nếu", "thế",
    "vậy", "ơi", "à", "ừ", "dạ", "vâng", "ạ", "đi", "chứ", "nhỉ", "nhá", "nhé", "quá", "rất", "thật", "còn",
    "nhưng", "tuy", "song", "hễ", "hầu", "bao", "gồm", "gì", "ai", "đâu", "nào", "sao", "bao nhiêu", "bấy nhiêu",
    "bởi", "do", "qua", "hay", "hoặc", "trở", "chỉ", "vẫn", "cứ", "luôn", "thường", "bao giờ", "lúc", "ngày",
    "tháng", "năm", "giờ", "phút", "giây", "đầu", "cuối", "về", "với", "đối với", "hướng", "cùng", "riêng",
    "chung", "hơn", "kém", "nhất", "số", "lượng", "thành", "phần", "toàn", "bộ", "tất cả", "hết", "vài",
    "ngày xửa ngày xưa", "ngày nảy ngày nay", "biết bao", "biết mấy", "ôi chao", "than ôi"
]

def extract_keywords_nlp(text: str) -> List[str]:
    """
    Trích xuất từ khóa từ văn bản sử dụng thư viện NLP (underthesea).
    Tập trung vào danh từ và danh từ riêng, loại bỏ stop words.
    """
    logger.info(f"Extracting keywords using NLP (underthesea) for text: {text[:50]}...")
    
    if not UNDERTHESEA_AVAILABLE:
        logger.warning("Underthesea is not available. Falling back to basic splitting.")
        # Fallback logic from previous version
        tokens = text.lower().split()
        keywords = [token for token in tokens if len(token) > 3 and token not in VIETNAMESE_STOP_WORDS]
        unique_keywords = list(set(keywords))
        logger.info(f"NLP (basic fallback) extracted keywords: {unique_keywords[:10]}")
        return unique_keywords[:10] # Lấy tối đa 10 từ khóa

    try:
        # 1. Tách từ và POS tagging
        # pos_tag trả về list các tuple (word, tag)
        tagged_words = pos_tag(text)
        
        extracted_keywords = []
        for word, tag in tagged_words:
            word_lower = word.lower()
            # 2. Lọc theo POS tag (ưu tiên danh từ, danh từ riêng)
            # N: Danh từ, Np: Danh từ riêng, Nc: Danh từ chỉ loại, Nu: Danh từ đơn vị
            # A: Tính từ - có thể xem xét thêm nếu muốn cụm từ
            if tag in ['N', 'Np', 'Nc', 'Nu']:
                # 3. Loại bỏ stop words và các từ quá ngắn
                if word_lower not in VIETNAMESE_STOP_WORDS and len(word_lower) > 2:
                    extracted_keywords.append(word_lower)
        
        # 4. Lấy danh sách duy nhất và giới hạn số lượng
        # Hiện tại đang trả về các từ đơn lẻ.
        # Để có cụm từ khóa (key phrases), cần logic phức tạp hơn để nhóm các từ liền kề dựa trên POS tags.
        # Ví dụ: (A* N+), (Np+ N*), ...
        unique_keywords = list(dict.fromkeys(extracted_keywords)) # Giữ thứ tự xuất hiện đầu tiên và đảm bảo duy nhất
        
        # Giới hạn số lượng từ khóa (ví dụ: top 10 hoặc 15)
        limit = 20
        final_keywords = unique_keywords[:limit]
        
        logger.info(f"NLP (underthesea) extracted keywords: {final_keywords}")
        return final_keywords
        
    except Exception as e:
        logger.error(f"Error during NLP keyword extraction with underthesea: {e}", exc_info=True)
        # Fallback an toàn nếu có lỗi với underthesea
        tokens = text.lower().split()
        keywords = [token for token in tokens if len(token) > 3 and token not in VIETNAMESE_STOP_WORDS]
        unique_keywords = list(set(keywords))
        logger.info(f"NLP (error fallback) extracted keywords: {unique_keywords[:10]}")
        return unique_keywords[:10]

async def extract_keywords_llm(text: str, llm_instance: Any) -> List[str]:
    """
    Trích xuất từ khóa từ văn bản sử dụng LLM.
    """
    logger.info(f"Extracting keywords using LLM for text: {text[:50]}...")
    # Đề xuất prompt chi tiết hơn, tập trung vào cụm từ khóa
    prompt = f"""
    Hãy trích xuất từ 5 đến 7 cụm từ khóa (key phrases) quan trọng nhất từ đoạn văn bản sau.
    Mỗi cụm từ khóa nên bao gồm từ 2 đến 4 từ, tập trung vào các khái niệm cốt lõi, thực thể nổi bật, hoặc thuật ngữ chuyên ngành có trong đoạn văn.
    Ưu tiên các cụm từ khóa mang tính mô tả cao và là danh từ hoặc cụm danh từ.
    Trả lời CHỈ bằng danh sách các cụm từ khóa, mỗi cụm từ khóa trên một dòng. Không thêm số thứ tự, gạch đầu dòng hay bất kỳ giải thích nào khác.

    Ví dụ:
    Văn bản mẫu: "Trường Đại học CMC công bố Quy chế Tổ chức và Hoạt động mới, áp dụng cho toàn thể giảng viên và sinh viên."
    Từ khóa:
    Đại học CMC
    Quy chế Tổ chức và Hoạt động
    giảng viên và sinh viên

    Văn bản cần trích xuất:
    ---
    {text}
    ---
    Từ khóa:
    """
    try:
        response_data = llm_instance.generate_content(prompt)
        
        content = ""
        if isinstance(response_data, dict):
            content = response_data.get("content", "")
        elif isinstance(response_data, str):
            content = response_data
        else:
            logger.warning("LLM response for keyword extraction has unexpected type.")
            return []

        keywords = [kw.strip() for kw in content.split('\n') if kw.strip() and kw.strip() != "Từ khóa:"] # Loại bỏ dòng "Từ khóa:" nếu LLM trả về
        logger.info(f"LLM extracted keywords: {keywords}")
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords with LLM: {e}", exc_info=True)
        return []

async def extract_keywords(
    text: str, 
    method: str = "nlp", 
    llm_instance: Any = None,
) -> List[str]:
    """
    Trích xuất từ khóa dựa trên phương thức được chọn.
    """
    if method == "llm":
        if not llm_instance:
            logger.error("LLM method selected for keyword extraction, but no LLM instance provided. Falling back to NLP.")
            return extract_keywords_nlp(text)
        return await extract_keywords_llm(text, llm_instance)
    elif method == "nlp":
        return extract_keywords_nlp(text)
    else:
        logger.warning(f"Unknown keyword extraction method: {method}. Defaulting to NLP.")
        return extract_keywords_nlp(text)

if __name__ == '__main__':
    # Ví dụ cách sử dụng (cần một LLM instance giả hoặc thật để test LLM part)
    sample_text_vn = "Trường Đại học CMC (mã trường: CMC) là một trường đại học tư thục tại Việt Nam, được thành lập vào năm 2011, trực thuộc Tập đoàn Công nghệ CMC. Trường đào tạo các ngành thuộc lĩnh vực công nghệ thông tin, kinh doanh và quản lý, thiết kế và truyền thông đa phương tiện. Sứ mệnh của trường là đào tạo nguồn nhân lực chất lượng cao, có khả năng thích ứng với sự thay đổi của thị trường lao động và đóng góp vào sự phát triển của đất nước."
    
    print("--- NLP Keywords (underthesea) ---")
    if UNDERTHESEA_AVAILABLE:
        nlp_kws = extract_keywords_nlp(sample_text_vn)
        print(nlp_kws)
    else:
        print("Underthesea not available, NLP test skipped for detailed extraction.")
        nlp_kws_fallback = extract_keywords_nlp(sample_text_vn) # Test fallback
        print("Fallback NLP:", nlp_kws_fallback)

    class MockLLM:
        def generate_content(self, prompt_text): # Đổi tên tham số để tránh trùng với biến prompt trong hàm
            print("\n--- LLM Prompt ---\n", prompt_text)
            # Giả lập LLM trả về, bao gồm cả dòng "Từ khóa:" mà prompt mới có thể tạo ra
            return {"content": "Từ khóa:\nĐại học CMC\nTập đoàn Công nghệ CMC\nCông nghệ thông tin\nThiết kế và truyền thông đa phương tiện\nNguồn nhân lực chất lượng cao", "usage": {"total_tokens": 50}}

    async def main_test():
        print("\n--- LLM Keywords (mocked) ---")
        llm_kws = await extract_keywords(sample_text_vn, method="llm", llm_instance=MockLLM())
        print(llm_kws)
        
        print("\n--- Default (NLP with underthesea if available) Keywords ---")
        default_kws = await extract_keywords(sample_text_vn) 
        print(default_kws)

    import asyncio
    asyncio.run(main_test()) 