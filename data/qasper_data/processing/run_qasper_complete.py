#!/usr/bin/env python3
"""
Script hoàn chỉnh để download và xử lý QASPER dataset
"""

import sys
import os

# Thêm current directory vào Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_qasper_simple import download_qasper_validation
from process_qasper_from_file import QASPERProcessorFromFile

def main():
    print("🚀 Starting Complete QASPER Dataset Processing...")
    print("=" * 55)
    
    try:
        # Bước 1: Download dữ liệu
        print("📥 Step 1: Downloading QASPER validation data...")
        data = download_qasper_validation()
        
        # Bước 2: Xử lý dữ liệu
        print("\n🔄 Step 2: Processing data into DOCX and QA format...")
        processor = QASPERProcessorFromFile()
        processor.process_dataset(limit=20)  # Xử lý 20 tài liệu đầu tiên
        
        print("\n✅ Complete processing finished successfully!")
        print("\n📁 Output structure:")
        print("data/")
        print("├── qasper_raw/")
        print("│   └── validation.json          # Raw QASPER data")
        print("└── qasper_processed/")
        print("    ├── docx_files/              # 20 file .docx (title + abstract + full_text)")
        print("    └── qa_data/")
        print("        └── qasper_qa_dataset.json  # QA data chuẩn hóa")
        
        print(f"\n🎯 Successfully processed 20 real QASPER documents!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
