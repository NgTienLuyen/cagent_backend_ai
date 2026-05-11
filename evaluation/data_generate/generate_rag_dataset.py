import json
import time
import google.generativeai as genai
import os # Để lấy API Key từ biến môi trường
import requests # Thêm import requests để gọi API

# --- Cấu hình API Key cho Gemini ---
# Tùy chọn 1: Lấy từ biến môi trường (khuyến nghị)
# Đảm bảo bạn đã đặt biến môi trường tên là GOOGLE_API_KEY trên hệ thống của bạn
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Tùy chọn 2: Thay thế trực tiếp YOUR_GEMINI_API_KEY bằng khóa API của bạn
genai.configure(api_key="AIzaSyBWbMFX1d2kE4LhRHI_6fH5QkT0dfMfMso") # THAY THẾ KHÓA API CỦA BẠN TẠI ĐÂY

# Khởi tạo mô hình Gemini
# Bạn có thể chọn model phù hợp, ví dụ: "gemini-pro" cho văn bản, "gemini-pro-vision" cho đa phương thức
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# --- Hàm để tải dữ liệu chunk từ API ---
def fetch_chunks_from_api():
    """
    Hàm để tải dữ liệu chunk từ API endpoint
    Thay đổi URL API theo endpoint thực tế của bạn
    """
    try:
        # Thay đổi URL này theo API endpoint thực tế của bạn
        api_url = "http://localhost:8000/api/chunks"  # Ví dụ endpoint
        
        print("Đang tải dữ liệu chunk từ API...")
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            chunks_data = response.json()
            print(f"Đã tải thành công {len(chunks_data.get('chunks', []))} chunks từ API")
            return chunks_data
        else:
            print(f"Lỗi API: HTTP {response.status_code}")
            print(f"Phản hồi: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Lỗi kết nối: Không thể kết nối đến API server")
        print("Hãy đảm bảo API server đang chạy và endpoint đúng")
        return None
    except requests.exceptions.Timeout:
        print("Lỗi timeout: API không phản hồi trong thời gian chờ")
        return None
    except Exception as e:
        print(f"Lỗi không xác định khi gọi API: {e}")
        return None

# --- Tải dữ liệu chunk từ API thay vì hardcode ---
print("Bắt đầu tải dữ liệu chunk từ API...")
raw_chunks_data = fetch_chunks_from_api()

# Nếu không thể tải từ API, sử dụng dữ liệu mẫu (fallback)
if raw_chunks_data is None:
    print("Không thể tải dữ liệu từ API. Sử dụng dữ liệu mẫu...")
    raw_chunks_data = {
      "chunks": [
        {
          "id": "f36f51f2-f283-4462-91bc-a8de1f33423a",
          "document_id": "ac477652-ba25-42cb-9dab-bd77b7e7c927",
          "chunk_text": "BỘ GIÁO DỤC VÀ ĐÀO TẠO TRƯỜNG ĐẠI HỌC CMC QUY CHẾ TỔ CHỨC VÀ HOẠT ĐỘNG TRƯỜNG ĐẠI HỌC CMC ( Ban hành kèm theo Quyết định số .. / 2023 / QĐ-ĐHCMC-HĐT của Chủ tịch Hội đồng Trường Đại học CMC ) Soạn thảo : Văn phòng Trường Hà Nội , tháng .. năm 2023 MỤC LỤC CHƯƠNG I NHỮNG QUY ĐỊNH CHUNG 5 Điều 1 . Phạm vi điều chỉnh và đối tượng áp dụng 5 Điều 2 . Giải thích từ ngữ 5 Điều 3 . Tên và trụ sở 6 Điều 4 . Địa vị pháp lý 6 Điều 5 . Nguyên tắc tổ chức và hoạt động 6 Điều 6 . Triết lý , sứ mạng , tâm nhin , giá trị cốt lõi cua Trương 7 Điều 7 . Nhiệm vụ và quyền hạn của Trường 7 Điều 8 . Cơ cấu tổ chức của Trường 8 Điều 9 . Quyền hạn và trách nhiệm của Nhà Đầu Tư 9 Điều 10 . Đại diện Nhà Đầu Tư 9 Điều 11 . Hội Nghị Nhà Đầu Tư 10 Điều 12 . Chuyển nhượng vốn gop 14 Điều 13 . Giây chưng nhân phân vôn gop 14 Điều 14 . Tăng hoăc giam Tổng Vôn Gop 14 Điều 15 . Ban Kiểm Soát 15 Điều 16 . Cơ cấu , thành phần và nhiệm kỳ của Hội Đồng Trường 16 Điều 17 . Chức năng , trách nhiệm và quyền hạn của Hội Đồng Trường 17 Điều 18 . Thủ tục bầu và công nhận Hội Đồng Trường 18 Điều 19 . Chủ Tịch Hội Đồng Trường 19 Điều 20 . Bãi nhiệm , miễn nhiệm thành viên Hội Đồng Trường 20 Điều 21 . Cuộc họp Hội Đồng Trường 21 Điều 22 . Lấy ý kiến thành viên Hội Đồng Trường bằng văn bản 22 Điều 23 . Hiệu Trưởng 23 Điều 24 . Phó Hiệu Trưởng 25 Điều 25 . Hội đồng khoa học và đào tạo 26 Điều 26 . Khoa 27 Điều 27 . Ban / phòng chức năng 28 Điều 28 . Thư viện 28 Điều 29 . Các tổ chức khoa học và công nghệ , tổ chức phục vụ đào tạo , cơ sở dịch vụ , doanh nghiệp , cơ sở kinh doanh 28 Điều 30 . Tổ chức Đảng và các tổ chức đoàn thể khác trong Trường 28 Điều 31 . Phân hiệu của Trường 29 CHƯƠNG III . HOẠT ĐỘNG GIÁO DỤC ĐÀO TẠO , KHOA HỌC , CÔNG NGHỆ VÀ HỢP TÁC 29 Điều 32 . Ngành nghề đào tạo 29 Điều 33 . Tuyển sinh 29 Điều 34 . Chương trình , giáo trình đào tạo 29 Điều 35 . Tổ chức và quản lý đào tạo 30 Điều 36 . Văn bằng , chứng chỉ 30 Điều 37 . Đảm bảo chất lượng giáo dục 30 Điều 38 . Hoạt động khoa học và công nghệ 31 Điều 39 .",
          "createTime": "2025-08-01T04:23:00.017431+00:00",
          "updated_at": "2025-08-01T04:23:00.017431+00:00",
          "isDelete": False
        },
        {
          "id": "16f162f1-9a01-45bc-8a3a-84aafd6032f8",
          "document_id": "ac477652-ba25-42cb-9dab-bd77b7e7c927",
          "chunk_text": "Hoạt động hợp tác 32 CHƯƠNG IV GIẢNG VIÊN VÀ NHÂN VIÊN 32 Điều 40 . Giảng viên 32 Điều 41 . Nhân viên 33 CHƯƠNG V. TÀI CHÍNH VÀ TÀI SẢN 33 Điều 42 . Nguồn tài chính của Trường 33 Điều 43 . Sử dụng nguồn tài chính của Trường 33 Điều 44 . Quản lý tài chính và tài sản 34 CHƯƠNG VI CHẾ ĐỘ THÔNG TIN , BÁO CÁO , THANH TRA , KIỂM TRA , KHEN THƯỞNG VÀ KỶ LUẬT 34 Điều 45 . Chế độ thông tin của Nhà Đầu Tư va Hội Đồng Trường 34 Điều 46 . Chế độ báo cáo của Ban giám hiệu 35 Điều 47 . Thanh tra , kiểm tra 35 Điều 48 . Khen thưởng 35 Điều 49 . Xử lý vi phạm 35 CHƯƠNG VII ĐIỀU KHOẢN THI HÀNH 36 Điều 50 . Hiệu lực thi hành 36 Phục lục 01 : Bảng phân quyền giữa Nhà Đầu Tư , Hội Đồng Trường và Hiệu Trưởng CHƯƠNG I NHỮNG QUY ĐỊNH CHUNG Điều Phạm vi điều chỉnh và đối tượng áp dụng Quy chế này quy định về tổ chức và hoạt động của Trường Đại học CMC ( sau đây gọi là Quy Chế ) , bao gồm : Tổ chức và Nhân sự ; Hoạt động Đào tạo , Khoa học , Công nghệ và Hợp tác ; Giảng viên , Nhân viên ; Tài chính và Tài sản ; Chế độ thông tin , Báo cáo , Thanh tra , Kiểm tra , Khen thưởng và Kỉ luật . Quy Chế này áp dụng đối với các Nhà Đầu Tư , các thành viên Hội Đồng Trường , các giảng viên , nhân viên của Trường , và các tổ chức , cá nhân khác có liên quan . Các quy định trong Quy Chế này là cơ sở tổ chức và hoạt động của Trường . Các quy định khác của Trường không được trái với Quy Chế này . Điều Giải thích từ ngữ Trong Quy Chế này , các thuật ngữ được viết hoa được hiểu như sau : Ban Giám Hiệu là ban giám hiệu của Trường , gồm Hiệu Trưởng , và ( các ) Phó Hiệu Trưởng . Giảng Viên Cơ Hữu / Cán Bộ Cơ Hữu là người lao động ký hợp đồng lao động có thời hạn 3 năm hoặc hợp đồng không xác định thời hạn theo Bộ luật Lao động , không là công chức hoặc viên chức nhà nước , không đang làm việc theo hợp đồng lao động có thời hạn từ 3 tháng trở lên với đơn vị sử dụng lao động khác ; do Trường trả lương và chi trả các khoản khác thuộc chế độ , chính sách đối với người lao động theo các quy định hiện hành .",
          "createTime": "2025-08-01T04:23:00.019516+00:00",
          "updated_at": "2025-08-01T04:23:00.019516+00:00",
          "isDelete": False
        }
        # ... có thể thêm các chunk mẫu khác nếu cần
      ]
    }

# Mẫu prompt cơ bản cho mỗi chunk
# Sử dụng {{ và }} để escape các dấu ngoặc nhọn trong JSON ví dụ
prompt_template = """Bạn là một chuyên gia trong việc tạo các cặp câu hỏi-trả lời chất lượng cao để đánh giá các hệ thống Sinh văn bản có tích hợp truy xuất (RAG - Retrieval-Augmented Generation). Nhiệm vụ của bạn là đọc kỹ đoạn văn bản được cung cấp và tạo ra **nhiều cặp câu hỏi-trả lời** (`question` và `ground_truth_answer`) liên quan đến các khía cạnh khác nhau hoặc các cách diễn đạt khác nhau của thông tin trong đoạn văn bản đó.

**Hướng dẫn chi tiết:**

1.  **Đọc và Hiểu `chunk_text`:** Phân tích kỹ nội dung của đoạn văn bản để nắm bắt thông tin chính và các chi tiết quan trọng.
2.  **Tạo Câu hỏi (`question`):**
    *   Đối với mỗi câu hỏi:
        *   Câu hỏi phải là một câu hỏi tự nhiên, giống như một người dùng thực tế sẽ hỏi.
        *   Mỗi câu hỏi phải hoàn toàn có thể được trả lời *trực tiếp và đầy đủ* từ thông tin có trong `chunk_text` được cung cấp.
        *   Đảm bảo câu hỏi không yêu cầu thông tin nằm ngoài `chunk_text`.
        *   **Tạo sự đa dạng:** Hãy cố gắng tạo các câu hỏi khác nhau về cùng một thông tin (diễn đạt lại), hoặc các câu hỏi tập trung vào các chi tiết/khía cạnh khác nhau trong cùng một đoạn văn bản. Nếu đoạn văn bản dài và chứa nhiều thông tin, hãy tạo nhiều câu hỏi hơn để bao phủ các điểm chính.
3.  **Trích xuất Câu trả lời chuẩn (`ground_truth_answer`):**
    *   Đối với mỗi câu hỏi, câu trả lời phải là **chính xác đoạn văn bản** từ `chunk_text` cung cấp câu trả lời cho câu hỏi đó.
    *   **Không được phép diễn giải, tóm tắt, hay thêm bất kỳ thông tin nào không có trong `chunk_text` gốc.** Mục tiêu là lấy ra \"ground truth\" (sự thật nền) trực tiếp từ nguồn.
    *   Nếu một câu hỏi có thể được trả lời bởi nhiều câu hoặc đoạn nhỏ liên tiếp trong chunk, hãy chọn phần chứa thông tin đầy đủ nhất.
4.  **Định dạng Đầu ra (JSON Array):**
    *   Bạn **PHẢI** trả về kết quả dưới dạng một **mảng (array) các đối tượng JSON**.
    *   Mỗi đối tượng trong mảng phải có hai trường: `\"question\"` và `\"ground_truth_answer\"`.
    *   **Mỗi chunk nên tạo ra từ 2 đến 5 cặp Q&A, tùy thuộc vào độ dài và độ phức tạp của chunk.** Nếu chunk quá ngắn hoặc chỉ có một thông tin duy nhất, có thể tạo ít hơn.

**Đoạn văn bản để xử lý (`chunk_text`):**
'''
{chunk_content}
'''

**Ví dụ về Định dạng Đầu ra mong muốn:**
```json
[
  {{
    \"question\": \"Câu hỏi 1 từ chunk này?\",
    \"ground_truth_answer\": \"Câu trả lời chính xác từ chunk cho câu hỏi 1.\"
  }},
  {{
    \"question\": \"Một câu hỏi khác hoặc biến thể từ chunk này?\",
    \"ground_truth_answer\": \"Câu trả lời chính xác từ chunk cho câu hỏi khác.\"
  }},
  {{
    \"question\": \"Câu hỏi thứ ba tập trung vào chi tiết khác?\",
    \"ground_truth_answer\": \"Phần văn bản liên quan đến câu hỏi thứ ba.\"
  }}
]
```
"""

# Danh sách để lưu trữ bộ dữ liệu đánh giá cuối cùng
evaluation_dataset = []
output_filename = "rag_evaluation_dataset.json"

# Kiểm tra nếu file output đã tồn tại và đọc dữ liệu cũ
if os.path.exists(output_filename):
    try:
        with open(output_filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                evaluation_dataset.extend(existing_data)
                print(f"Đã tải {len(existing_data)} cặp Q&A hiện có từ {output_filename}.")
            else:
                print(f"Cảnh báo: Tệp {output_filename} không chứa danh sách JSON hợp lệ. Bắt đầu với bộ dữ liệu trống.")
    except json.JSONDecodeError:
        print(f"Cảnh báo: Tệp {output_filename} không phải JSON hợp lệ. Bắt đầu với bộ dữ liệu trống.")
    except Exception as e:
        print(f"Lỗi khi đọc tệp {output_filename}: {e}. Bắt đầu với bộ dữ liệu trống.")

# Kiểm tra nếu raw_chunks_data.chunks không rỗng
if not raw_chunks_data["chunks"]:
    print("Lỗi: Không tìm thấy dữ liệu chunk. Vui lòng dán dữ liệu chunk của bạn vào biến 'raw_chunks_data' trong script này.")
else:
    # Lặp qua từng chunk trong dữ liệu của bạn
    for i, chunk in enumerate(raw_chunks_data["chunks"]):
        chunk_id = chunk["id"]
        chunk_text = chunk["chunk_text"]
        document_id = chunk.get("document_id", "unknown_document")

        print(f"[{i+1}/{len(raw_chunks_data['chunks'])}] Đang xử lý chunk: {chunk_id} (tài liệu: {document_id})...")

        formatted_prompt = prompt_template.format(chunk_content=chunk_text)

        try:
            response = model.generate_content(
                formatted_prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            llm_output_text = response.text

            qa_pairs_list = json.loads(llm_output_text)

            if isinstance(qa_pairs_list, list):
                for qa_pair in qa_pairs_list:
                    qa_entry = {
                        "question": qa_pair.get("question", "N/A"),
                        "ground_truth_answer": qa_pair.get("ground_truth_answer", "N/A"),
                        "relevant_contexts": [chunk_text],  # Lưu nội dung text
                        "relevant_chunk_ids": [chunk_id],  # Giữ lại ID để tham khảo
                        "source_document_id": document_id
                    }
                    evaluation_dataset.append(qa_entry)
                print(f"  -> Đã tạo {len(qa_pairs_list)} cặp Q&A thành công.")
            else:
                print(f"  -> Lỗi: LLM không trả về danh sách JSON hợp lệ cho chunk {chunk_id}. Phản hồi thô: {llm_output_text}")

        except json.JSONDecodeError:
            print(f"  -> Lỗi: LLM không trả về JSON hợp lệ cho chunk {chunk_id}. Phản hồi thô: {llm_output_text}")
        except Exception as e:
            print(f"  -> Đã xảy ra lỗi khi gọi LLM cho chunk {chunk_id}: {e}")

        time.sleep(0.5)

    # Lưu bộ dữ liệu đánh giá ra file JSON (bao gồm cả dữ liệu cũ và mới)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(evaluation_dataset, f, ensure_ascii=False, indent=2)

    print("\n======================================")
    print("Quá trình tạo bộ dữ liệu đã hoàn tất.")
    print(f"Đã tạo thành công bộ dữ liệu đánh giá và lưu vào: {output_filename}")
    print(f"Tổng số cặp Q&A được tạo: {len(evaluation_dataset)}")
    print("======================================")