#!/usr/bin/env python3
"""
Script đơn giản để chuyển đổi dataset NarrativeQA thành file Word
Sử dụng trực tiếp với load_dataset
"""

from datasets import load_dataset
from data.convert_dataset_to_word_files import convert_dataset_to_word_files
import os

def main():
    """Hàm chính"""
    print("🚀 CHUYỂN ĐỔI DATASET NARRATIVEQA THÀNH FILE WORD")
    print("=" * 60)
    
    # Cấu hình
    output_dir = input("📂 Nhập thư mục output (mặc định: narrativeqa_word_files): ").strip()
    if not output_dir:
        output_dir = "narrativeqa_word_files"
    
    max_items = input("🎯 Nhập số items tối đa để chuyển đổi (để trống để chuyển đổi tất cả): ").strip()
    if max_items:
        try:
            max_items = int(max_items)
        except ValueError:
            print("❌ Số items không hợp lệ, sẽ chuyển đổi tất cả")
            max_items = None
    
    # Load dataset NarrativeQA
    print("\n🔄 Đang load dataset NarrativeQA...")
    try:
        # Load validation split (nhỏ hơn để test)
        dataset = load_dataset("deepmind/narrativeqa", split="validation")
        print(f"✅ Đã load dataset NarrativeQA validation với {len(dataset)} items")
        
        # Giới hạn số items nếu cần
        if max_items and max_items < len(dataset):
            dataset = dataset.select(range(max_items))
            print(f"🎯 Sẽ chuyển đổi {len(dataset)} items đầu tiên")
        
        # Hiển thị cấu trúc dataset
        print(f"\n📊 Cấu trúc dataset:")
        print(f"   - Số items: {len(dataset)}")
        print(f"   - Các trường: {list(dataset.features.keys())}")
        
        # Hiển thị ví dụ đầu tiên
        print(f"\n📝 Ví dụ item đầu tiên:")
        first_item = dataset[0]
        for key, value in first_item.items():
            if isinstance(value, str):
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {type(value)} - {value}")
        
        # Xác nhận tiếp tục
        confirm = input(f"\n⏳ Nhấn Enter để bắt đầu chuyển đổi {len(dataset)} items...")
        
        # Chạy chuyển đổi
        files = convert_dataset_to_word_files(dataset, output_dir, "NarrativeQA")
        
        print(f"\n🎉 Hoàn thành! Đã tạo {len(files)} file Word")
        print(f"📁 Output directory: {output_dir}")
        print(f"🗂️ Mapping file: {os.path.join(output_dir, 'file_mapping.json')}")
        
        # Hướng dẫn sử dụng
        print(f"\n💡 Bước tiếp theo:")
        print(f"   1. Upload các file Word trong {output_dir}/ vào hệ thống RAG")
        print(f"   2. Thực hiện chunking và embedding")
        print(f"   3. Sử dụng file mapping để đánh giá RAG system")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Đảm bảo bạn đã cài đặt thư viện datasets:")
        print("   pip install datasets")

if __name__ == "__main__":
    main()

