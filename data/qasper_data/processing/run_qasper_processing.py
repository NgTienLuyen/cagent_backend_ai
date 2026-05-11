#!/usr/bin/env python3
"""
Script chạy đơn giản để xử lý QASPER dataset
"""

import sys
import os

# Thêm current directory vào Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_qasper_dataset import QASPERProcessor

def main():
    print("🚀 Starting QASPER Dataset Processing...")
    print("=" * 50)
    
    try:
        processor = QASPERProcessor()
        processor.process_dataset(limit=20)  # Chỉ xử lý 20 tài liệu đầu tiên
        
        print("\n✅ Processing completed successfully!")
        print("\n📁 Output structure:")
        print("data/qasper_processed/")
        print("├── docx_files/          # 20 file .docx (title + abstract + full_text)")
        print("└── qa_data/            # File JSON chứa QA data chuẩn hóa")
        print("    └── qasper_qa_dataset.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
