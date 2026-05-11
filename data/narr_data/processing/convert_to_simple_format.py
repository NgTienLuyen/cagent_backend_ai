import json
import os
from datasets import load_dataset
from collections import defaultdict
import uuid

def convert_to_simple_format(dataset, output_file="qa_dataset.json"):
    """
    Chuyển đổi dataset từ cấu trúc phức tạp sang cấu trúc đơn giản
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
    
    # Xử lý tất cả các tài liệu
    all_docs = list(docs_grouped.items())
    
    # Chuyển đổi sang format đơn giản
    simple_dataset = []
    total_records = 0
    
    for doc_id, items in all_docs:
        print(f"📝 Đang xử lý tài liệu {doc_id[:20]}... với {len(items)} Q&A pairs...")
        
        # Lấy nội dung tài liệu (ưu tiên summary.text)
        document_content = ""
        if items and 'document' in items[0]:
            doc_info = items[0]['document']
            if 'summary' in doc_info and 'text' in doc_info['summary']:
                document_content = doc_info['summary']['text']
            elif 'text' in doc_info:
                document_content = doc_info['text']
            elif 'full_text' in doc_info:
                document_content = doc_info['full_text']
        
        # Tạo chunk ID cho tài liệu này
        chunk_id = str(uuid.uuid4())
        
        # Chuyển đổi từng Q&A pair
        for item in items:
            # Lấy câu hỏi
            question = ""
            if 'question' in item and 'text' in item['question']:
                question = item['question']['text']
            elif 'question' in item:
                question = str(item['question'])
            
            # Lấy câu trả lời
            ground_truth_answer = ""
            if 'answers' in item and item['answers']:
                if isinstance(item['answers'], list):
                    # Lấy câu trả lời đầu tiên
                    first_answer = item['answers'][0]
                    if isinstance(first_answer, dict) and 'text' in first_answer:
                        ground_truth_answer = first_answer['text']
                    else:
                        ground_truth_answer = str(first_answer)
                else:
                    ground_truth_answer = str(item['answers'])
            elif 'answer' in item:
                ground_truth_answer = str(item['answer'])
            
            # Tạo record mới với format đơn giản
            simple_record = {
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "relevant_contexts": [document_content] if document_content else [],
                "relevant_chunk_ids": [chunk_id] if chunk_id else [],
                "source_document_id": doc_id
            }
            
            simple_dataset.append(simple_record)
            total_records += 1
    
    # Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simple_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Hoàn thành! Đã tạo {output_file}")
    print(f"📊 Tổng số bản ghi: {total_records}")
    print(f"📄 Số tài liệu: {len(all_docs)}")
    
    # In thống kê
    print(f"\n📈 Thống kê:")
    for doc_id, items in all_docs:
        first_item = items[0]
        doc_info = first_item.get('document', {})
        kind = doc_info.get('kind', 'unknown')
        word_count = doc_info.get('word_count', 0)
        
        print(f"   📄 {doc_id[:20]}... ({kind}): {len(items)} Q&A pairs, {word_count} words")
    
    # In mẫu dữ liệu
    if simple_dataset:
        print(f"\n📋 Mẫu dữ liệu đầu tiên:")
        sample = simple_dataset[0]
        print(f"   ❓ Question: {sample['question'][:100]}...")
        print(f"   💡 Answer: {sample['ground_truth_answer'][:100]}...")
        print(f"   📄 Context length: {len(sample['relevant_contexts'][0]) if sample['relevant_contexts'] else 0} chars")
        print(f"   🆔 Chunk ID: {sample['relevant_chunk_ids'][0] if sample['relevant_chunk_ids'] else 'N/A'}")
        print(f"   📁 Source: {sample['source_document_id'][:20]}...")
    
    return simple_dataset

def main():
    print("🚀 CHUYỂN ĐỔI DATASET SANG FORMAT ĐƠN GIẢN")
    print("=" * 60)
    print("📝 Tạo dataset với cấu trúc: question, answer, context, chunk_id, source_id")
    print("=" * 60)
    
    # Load dataset
    try:
        print("🔄 Đang load dataset...")
        dataset = load_dataset("deepmind/narrativeqa", split="validation")
        print(f"✅ Đã load: {len(dataset)} items")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return
    
    # Chuyển đổi
    output_file = input("📁 Tên file output (mặc định: simple_qa_dataset.json): ").strip()
    if not output_file:
        output_file = "simple_qa_dataset.json"
    
    try:
        simple_data = convert_to_simple_format(dataset, output_file)
        print(f"\n✅ Dataset đã được chuyển đổi và lưu vào: {output_file}")
        print("💡 Format mới: question, ground_truth_answer, relevant_contexts, relevant_chunk_ids, source_document_id")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
