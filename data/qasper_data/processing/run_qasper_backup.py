#!/usr/bin/env python3
"""
Script chạy backup để test với dữ liệu mẫu
"""

import sys
import os

# Thêm current directory vào Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_qasper_dataset_backup import QASPERProcessorBackup

def main():
    print("🚀 Starting QASPER Dataset Processing (BACKUP with Sample Data)...")
    print("=" * 60)
    
    try:
        processor = QASPERProcessorBackup()
        processor.process_dataset(limit=5)  # Test với 5 tài liệu mẫu
        
        print("\n✅ Processing completed successfully!")
        print("\n📁 Output structure:")
        print("data/qasper_processed/")
        print("├── docx_files/          # 5 file .docx (sample data)")
        print("└── qa_data/            # File JSON chứa QA data chuẩn hóa")
        print("    └── qasper_qa_dataset.json")
        print("\n💡 This is sample data for testing. To use real QASPER data,")
        print("   you need to fix the Hugging Face dataset loading issue.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
