import json
import pandas as pd
import os

def quick_calculate_averages():
    """Tính nhanh điểm trung bình từ file kết quả"""
    
    # Tìm file kết quả gần nhất
    possible_files = [
        "rag_evaluation_results_extra_metrics.json",
        "rag_evaluation_results_extra_metrics_english.json", 
        "rag_evaluation_results.json"
    ]
    
    results_file = None
    for file_name in possible_files:
        file_path = os.path.join("..", file_name)
        if os.path.exists(file_path):
            results_file = file_path
            break
    
    if not results_file:
        print("❌ Không tìm thấy file kết quả nào!")
        return
    
    print(f"📁 Đang xử lý file: {results_file}")
    
    # Load kết quả
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✅ Đã load {len(results)} kết quả")
    except Exception as e:
        print(f"❌ Lỗi khi load file: {e}")
        return
    
    # Tính điểm trung bình
    metrics = {}
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float)) and key not in ['question_id', 'retrieved_contexts_count']:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
    
    # Tạo bảng kết quả
    summary_data = []
    for metric, values in metrics.items():
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        summary_data.append({
            'Metric': metric,
            'Average': round(avg, 4),
            'Min': round(min_val, 4),
            'Max': round(max_val, 4),
            'Count': len(values)
        })
    
    # Sắp xếp theo tên metric
    summary_data.sort(key=lambda x: x['Metric'])
    
    # Tạo DataFrame và lưu CSV
    df = pd.DataFrame(summary_data)
    output_file = "quick_average_scores.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n📊 KẾT QUẢ TÍNH TOÁN:")
    print(f"📈 Số metrics: {len(metrics)}")
    print(f"📝 Số kết quả: {len(results)}")
    
    # In điểm trung bình
    print(f"\n🏆 ĐIỂM SỐ TRUNG BÌNH:")
    for row in summary_data:
        print(f"  {row['Metric']}: {row['Average']:.4f} [{row['Min']:.4f} - {row['Max']:.4f}]")
    
    print(f"\n💾 Đã lưu kết quả vào: {output_file}")

if __name__ == "__main__":
    quick_calculate_averages()

