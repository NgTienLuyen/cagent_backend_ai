import json
import os
from datasets import load_dataset
from collections import defaultdict

def simple_extract_first_5(dataset, output_file="first_5_documents.json"):
    """
    Trích xuất đơn giản: lấy tất cả bản ghi của 5 tài liệu đầu tiên
    """
    print(f"✅ Đã load dataset với {len(dataset)} items")
    
    # Nhóm theo document.id
    print("🔄 Đang nhóm theo document ID...")
    docs_grouped = defaultdict(list)
    
    for item in dataset:
        # Lấy document ID
        if 'document' in item and 'id' in item['document']:
            doc_id = item['document']['id']
        else:
            doc_id = "unknown"
        
        docs_grouped[doc_id].append(item)
    
    print(f"📊 Tìm thấy {len(docs_grouped)} tài liệu duy nhất")
    
    # Lấy 5 tài liệu đầu tiên
    first_5_docs = list(docs_grouped.items())[:5]
    
    # Tạo dataset mới
    extracted_data = []
    total_records = 0
    
    for doc_id, items in first_5_docs:
        print(f"📝 Tài liệu {doc_id[:20]}...: {len(items)} bản ghi")
        
        # Thêm tất cả bản ghi của tài liệu này
        for item in items:
            extracted_data.append(item)
            total_records += 1
    
    # Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Hoàn thành! Đã tạo {output_file}")
    print(f"📊 Tổng số bản ghi: {total_records}")
    print(f"📄 Số tài liệu: {len(first_5_docs)}")
    
    # In thống kê
    print(f"\n📈 Thống kê:")
    for doc_id, items in first_5_docs:
        first_item = items[0]
        doc_info = first_item.get('document', {})
        kind = doc_info.get('kind', 'unknown')
        word_count = doc_info.get('word_count', 0)
        
        print(f"   📄 {doc_id[:20]}... ({kind}): {len(items)} Q&A pairs, {word_count} words")
    
    return extracted_data

def main():
    print("🚀 TRÍCH XUẤT 5 TÀI LIỆU ĐẦU TIÊN")
    print("=" * 50)
    
    # Load dataset
    try:
        print("🔄 Đang load dataset...")
        dataset = load_dataset("deepmind/narrativeqa", split="validation")
        print(f"✅ Đã load: {len(dataset)} items")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return
    
    # Trích xuất
    output_file = input("📁 Tên file output (mặc định: first_5_documents.json): ").strip()
    if not output_file:
        output_file = "first_5_documents.json"
    
    try:
        simple_extract_first_5(dataset, output_file)
        print(f"\n✅ Dataset đã được lưu vào: {output_file}")
        print("💡 Bạn có thể mở file này để xem cấu trúc dữ liệu!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
