import json
import pandas as pd
import os
from typing import Dict, List
import numpy as np

def load_english_results(file_path: str) -> List[Dict]:
    """Load kết quả đánh giá từ file English JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✅ Đã load {len(results)} kết quả từ {file_path}")
        return results
    except Exception as e:
        print(f"❌ Lỗi khi load file {file_path}: {e}")
        return []

def calculate_average_scores(results: List[Dict]) -> Dict:
    """Tính toán điểm số trung bình cho tất cả metrics"""
    if not results:
        return {}
    
    # Lấy tất cả các keys có thể có trong results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # Loại bỏ các keys không phải là số
    numeric_keys = []
    for key in all_keys:
        if key in ['question_id', 'question', 'generated_answer', 'ground_truth', 'retrieved_contexts_count']:
            continue
        
        # Kiểm tra xem key có chứa giá trị số không
        sample_values = [result.get(key) for result in results if result.get(key) is not None]
        if sample_values and all(isinstance(v, (int, float)) for v in sample_values):
            numeric_keys.append(key)
    
    print(f"📊 Tìm thấy {len(numeric_keys)} metrics số: {numeric_keys}")
    
    # Tính toán điểm trung bình
    averages = {}
    for key in numeric_keys:
        values = [result.get(key) for result in results if result.get(key) is not None]
        if values:
            avg_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)
            
            averages[key] = {
                'average': round(avg_value, 4),
                'std': round(std_value, 4),
                'min': round(min_value, 4),
                'max': round(max_value, 4),
                'count': len(values)
            }
    
    return averages

def create_summary_dataframe(averages: Dict) -> pd.DataFrame:
    """Tạo DataFrame tóm tắt kết quả"""
    if not averages:
        return pd.DataFrame()
    
    # Tạo DataFrame
    summary_data = []
    for metric, stats in averages.items():
        summary_data.append({
            'Metric': metric,
            'Average': stats['average'],
            'Std Dev': stats['std'],
            'Min': stats['min'],
            'Max': stats['max'],
            'Count': stats['count']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Sắp xếp theo tên metric
    df = df.sort_values('Metric')
    
    return df

def analyze_metrics_by_category(averages: Dict) -> Dict:
    """Phân tích metrics theo nhóm"""
    categories = {
        'Core Quality': ['faithfulness_score', 'relevancy_score', 'context_recall_score', 'similarity_score'],
        'Retrieval': ['precision@k', 'recall@k', 'mrr', 'ndcg'],
        'Text Quality': ['bleu', 'rouge-1', 'rouge-2', 'rouge-l'],
        'Semantic': ['semantic_similarity'],
        'Overall': ['overall_score']
    }
    
    category_averages = {}
    for category, metrics in categories.items():
        category_scores = {}
        for metric in metrics:
            if metric in averages:
                category_scores[metric] = averages[metric]['average']
        
        if category_scores:
            category_avg = np.mean(list(category_scores.values()))
            category_averages[category] = {
                'metrics': category_scores,
                'category_average': round(category_avg, 4)
            }
    
    return category_averages

def save_results_to_csv(df: pd.DataFrame, output_path: str):
    """Lưu kết quả vào file CSV"""
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 Đã lưu kết quả vào: {output_path}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file CSV: {e}")

def main():
    """Hàm chính - xử lý file English results"""
    print("🚀 Bắt đầu xử lý file English evaluation results...")
    print("=" * 60)
    
    # Đường dẫn đến file English results
    english_file_path = "kq/rag_evaluation_results_extra_metrics_english.json"
    
    if not os.path.exists(english_file_path):
        print(f"❌ Không tìm thấy file: {english_file_path}")
        return
    
    # Load kết quả
    results = load_english_results(english_file_path)
    
    if not results:
        print("❌ Không có kết quả nào để xử lý")
        return
    
    print(f"\n📊 Tổng cộng: {len(results)} kết quả đánh giá")
    print("=" * 60)
    
    # Tính toán điểm trung bình
    averages = calculate_average_scores(results)
    
    if not averages:
        print("❌ Không thể tính toán điểm trung bình")
        return
    
    # Tạo DataFrame tóm tắt
    summary_df = create_summary_dataframe(averages)
    
    # Lưu kết quả chính
    output_path = "english_evaluation_average_scores.csv"
    save_results_to_csv(summary_df, output_path)
    
    # Phân tích theo nhóm
    category_analysis = analyze_metrics_by_category(averages)
    
    # Tạo bảng tóm tắt theo nhóm
    category_summary = []
    for category, data in category_analysis.items():
        category_summary.append({
            'Category': category,
            'Category Average': data['category_average'],
            'Metrics Count': len(data['metrics']),
            'Best Metric': max(data['metrics'].items(), key=lambda x: x[1])[0] if data['metrics'] else 'N/A',
            'Best Score': max(data['metrics'].values()) if data['metrics'] else 0
        })
    
    category_df = pd.DataFrame(category_summary)
    category_output_path = "english_evaluation_category_summary.csv"
    save_results_to_csv(category_df, category_output_path)
    
    # In kết quả tóm tắt
    print("\n" + "=" * 60)
    print("🎉 KẾT QUẢ TÍNH TOÁN ĐIỂM SỐ TRUNG BÌNH (ENGLISH)")
    print("=" * 60)
    
    print(f"📊 Tổng số kết quả: {len(results)}")
    print(f"📈 Số metrics: {len(averages)}")
    
    # In điểm trung bình của các metrics quan trọng
    important_metrics = ['faithfulness_score', 'relevancy_score', 'context_recall_score', 
                        'similarity_score', 'overall_score', 'semantic_similarity']
    
    print("\n🏆 ĐIỂM SỐ TRUNG BÌNH CÁC METRICS QUAN TRỌNG:")
    for metric in important_metrics:
        if metric in averages:
            stats = averages[metric]
            print(f"  {metric}: {stats['average']:.4f} (±{stats['std']:.4f}) [{stats['min']:.4f} - {stats['max']:.4f}]")
    
    # In phân tích theo nhóm
    print("\n📋 PHÂN TÍCH THEO NHÓM:")
    for category, data in category_analysis.items():
        print(f"  {category}: {data['category_average']:.4f}")
    
    print("\n✅ Hoàn thành! Kiểm tra các file CSV đã tạo:")
    print(f"  📊 Chi tiết: {output_path}")
    print(f"  📋 Tóm tắt nhóm: {category_output_path}")

if __name__ == "__main__":
    main()

