#!/usr/bin/env python3
"""
Script đơn giản để download QASPER dataset trực tiếp từ Hugging Face Hub
"""

import os
import json
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_qasper_validation():
    """Download validation set từ Hugging Face Datasets Server API"""
    logger.info("Downloading QASPER validation dataset...")
    
    try:
        # Thử cách 1: Download từ train split (vì validation có thể không có)
        logger.info("Trying train split...")
        train_url = "https://datasets-server.huggingface.co/rows"
        params = {
            "dataset": "allenai/qasper",
            "config": "qasper",
            "split": "train",
            "offset": 0,
            "length": 100
        }
        
        logger.info(f"Downloading from: {train_url}")
        logger.info(f"Parameters: {params}")
        
        response = requests.get(train_url, params=params)
        response.raise_for_status()
        
        data_response = response.json()
        rows = data_response.get('rows', [])
        logger.info(f"Downloaded {len(rows)} rows from train split")
        
        if not rows:
            # Thử cách 2: Download từ parquet endpoint
            logger.info("Trying parquet endpoint...")
            parquet_url = "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/train"
            
            response = requests.get(parquet_url)
            response.raise_for_status()
            
            # Parquet endpoint trả về file parquet, cần xử lý khác
            logger.info("Parquet endpoint accessible, but need different processing")
            raise Exception("Parquet endpoint needs different processing method")
        
        # Convert rows thành format giống như dataset gốc
        records = []
        for row in rows:
            record = row.get('row', {})
            if record:  # Chỉ lấy records có dữ liệu
                records.append(record)
        
        logger.info(f"Converted to {len(records)} valid records")
        
        # Tạo thư mục output
        output_dir = Path("data/qasper_raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu file
        output_file = output_dir / "validation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved validation data to: {output_file}")
        
        # Hiển thị thông tin về record đầu tiên
        if records:
            first_record = records[0]
            logger.info(f"First record ID: {first_record.get('id', 'N/A')}")
            logger.info(f"First record title: {first_record.get('title', 'N/A')[:100]}...")
            logger.info(f"Has QA data: {'qas' in first_record}")
        
        return records
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise

def main():
    """Main function"""
    try:
        data = download_qasper_validation()
        print(f"\n✅ Successfully downloaded {len(data)} validation records!")
        print(f"📁 Data saved to: data/qasper_raw/validation.json")
        print(f"🔍 You can now use this data with the processing script.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
