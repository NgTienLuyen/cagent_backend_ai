import json
import requests
import os
import google.generativeai as genai
from typing import List, Dict
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# --- Cấu hình ---
GOOGLE_API_KEY = "AIzaSyCA10pkwfBjhPCX_FdBBO6Ig7twxwQPgII"
genai.configure(api_key=GOOGLE_API_KEY)

RAG_API_URL = "http://localhost:8000/api/query/"
EVAL_DATASET_PATH = "C:/Users/DHS/Desktop/InternCMC/CMC_chatbot/aiagent_ai_backend/data/simple_qa_dataset.json"
DELAY_BETWEEN_QUESTIONS = 120

# --- Lưu ý: Đã sửa lỗi BLEU và ROUGE = 0 ---
# Vấn đề: BLEU và ROUGE phân biệt ký tự hoa thường
# Giải pháp: Chuẩn hóa text về lowercase trước khi so sánh
# Ví dụ: "THE ACTOR" ≠ "the actor" → "the actor" = "the actor" ✅

# --- Semantic similarity helper ---
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_semantic_match(a, b, threshold=0.5):
    emb_a = model.encode(a)
    emb_b = model.encode(b)
    return util.cos_sim(emb_a, emb_b).item() > threshold

# --- Metrics mới với semantic matching ---
def precision_at_k_semantic(retrieved: List[str], relevant: List[str], k: int, threshold=0.5) -> float:
    retrieved_k = retrieved[:k]
    hits = 0
    for doc in retrieved_k:
        if any(is_semantic_match(doc, rel, threshold) for rel in relevant):
            hits += 1
    return hits / k if k > 0 else 0.0

def recall_at_k_semantic(retrieved: List[str], relevant: List[str], k: int, threshold=0.3) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    matched = set()
    for rel in relevant:
        if any(is_semantic_match(doc, rel, threshold) for doc in retrieved_k):
            matched.add(rel)
    return len(matched) / len(relevant)

def mrr_semantic(retrieved: List[str], relevant: List[str], threshold=0.5) -> float:
    for rank, doc in enumerate(retrieved, 1):
        if any(is_semantic_match(doc, rel, threshold) for rel in relevant):
            return 1.0 / rank
    return 0.0

def ndcg_semantic(retrieved: List[str], relevant: List[str], k: int, threshold=0.5) -> float:
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        if any(is_semantic_match(doc, rel, threshold) for rel in relevant):
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def bleu_score(reference: str, candidate: str) -> float:
    """Tính BLEU score với text đã được chuẩn hóa"""
    try:
        import re
        
        # Chuẩn hóa text: chuyển về lowercase, strip, và loại bỏ dấu câu
        def clean_text(text):
            # Loại bỏ dấu câu và ký tự đặc biệt, chỉ giữ lại chữ cái và số
            cleaned = re.sub(r'[^\w\s]', '', text.lower().strip())
            return cleaned
        
        reference_normalized = clean_text(reference)
        candidate_normalized = clean_text(candidate)
        
        # Tách từ
        reference_tokens = [reference_normalized.split()]
        candidate_tokens = candidate_normalized.split()
        
        # Sử dụng smoothing để tránh BLEU = 0
        from nltk.translate.bleu_score import SmoothingFunction
        smoothing = SmoothingFunction().method1
        
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
    except Exception as e:
        print(f"Lỗi khi tính BLEU score: {e}")
        return 0.0

def rouge_score(reference: str, candidate: str) -> Dict:
    """Tính ROUGE score với text đã được chuẩn hóa"""
    try:
        import re
        rouge = Rouge()
        
        # Chuẩn hóa text: chuyển về lowercase, strip, và loại bỏ dấu câu
        def clean_text(text):
            # Loại bỏ dấu câu và ký tự đặc biệt, chỉ giữ lại chữ cái và số
            cleaned = re.sub(r'[^\w\s]', '', text.lower().strip())
            return cleaned
        
        reference_normalized = clean_text(reference)
        candidate_normalized = clean_text(candidate)
        
        # Tính ROUGE score
        scores = rouge.get_scores(candidate_normalized, reference_normalized)
        
        if scores and len(scores) > 0:
            return scores[0]
        else:
            # Trả về giá trị mặc định nếu không tính được
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }
    except Exception as e:
        print(f"Lỗi khi tính ROUGE score: {e}")
        # Trả về giá trị mặc định nếu có lỗi
        return {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }

def semantic_similarity(text1: str, text2: str) -> float:
    """Tính semantic similarity với text đã được chuẩn hóa"""
    try:
        import re
        
        # Chuẩn hóa text: chuyển về lowercase, strip, và loại bỏ dấu câu
        def clean_text(text):
            # Loại bỏ dấu câu và ký tự đặc biệt, chỉ giữ lại chữ cái và số
            cleaned = re.sub(r'[^\w\s]', '', text.lower().strip())
            return cleaned
        
        text1_normalized = clean_text(text1)
        text2_normalized = clean_text(text2)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode([text1_normalized])[0]
        emb2 = model.encode([text2_normalized])[0]
        return float(cosine_similarity([emb1], [emb2])[0][0])
    except Exception as e:
        print(f"Lỗi khi tính semantic similarity: {e}")
        return 0.0

def debug_text_normalization(reference: str, candidate: str):
    """Debug: hiển thị quá trình chuẩn hóa text"""
    import re
    
    # Debug text normalization removed for production
    
    # Chuẩn hóa text: lowercase, strip, và loại bỏ dấu câu
    def clean_text(text):
        # Loại bỏ dấu câu và ký tự đặc biệt, chỉ giữ lại chữ cái và số
        cleaned = re.sub(r'[^\w\s]', '', text.lower().strip())
        return cleaned
    
    reference_cleaned = clean_text(reference)
    candidate_cleaned = clean_text(candidate)
    
    print(f"  Reference (chuẩn hóa): '{reference_cleaned}'")
    print(f"  Candidate (chuẩn hóa): '{candidate_cleaned}'")
    
    # Tính overlap từ sau khi đã làm sạch
    ref_words = set(reference_cleaned.split())
    cand_words = set(candidate_cleaned.split())
    overlap = ref_words.intersection(cand_words)
    
    print(f"  Reference words: {list(ref_words)}")
    print(f"  Candidate words: {list(cand_words)}")
    print(f"  Overlap words: {list(overlap)}")
    print(f"  Overlap ratio: {len(overlap)}/{len(ref_words)} = {len(overlap)/len(ref_words):.3f}")
    print("=" * 50)

# Thêm hàm tính Recall@K cho nhiều giá trị K
def recall_at_k_multiple_k(retrieved: List[str], relevant: List[str], k_values: List[int] = [1, 2, 3, 4, 5], threshold=0.2) -> Dict[int, float]:
    """Tính Recall@K cho nhiều giá trị K khác nhau"""
    results = {}
    for k in k_values:
        results[k] = recall_at_k_semantic(retrieved, relevant, k, threshold)
    return results

# Thêm hàm tính BLEU-1 và BLEU-4
def bleu_scores_multiple_n(reference: str, candidate: str, n_values: List[int] = [1, 4]) -> Dict[int, float]:
    """Tính BLEU score cho nhiều giá trị n-gram khác nhau"""
    try:
        import re
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Chuẩn hóa text
        def clean_text(text):
            cleaned = re.sub(r'[^\w\s]', '', text.lower().strip())
            return cleaned
        
        reference_normalized = clean_text(reference)
        candidate_normalized = clean_text(candidate)
        
        # Tách từ
        reference_tokens = [reference_normalized.split()]
        candidate_tokens = candidate_normalized.split()
        
        # Sử dụng smoothing
        smoothing = SmoothingFunction().method1
        
        results = {}
        for n in n_values:
            if n == 1:
                # BLEU-1: chỉ tính unigram
                weights = (1.0, 0.0, 0.0, 0.0)
            elif n == 4:
                # BLEU-4: tính từ unigram đến 4-gram
                weights = (0.25, 0.25, 0.25, 0.25)
            else:
                # BLEU-n: tính từ unigram đến n-gram
                weights = tuple([1.0/n] * n) + tuple([0.0] * (4-n))
            
            score = sentence_bleu(reference_tokens, candidate_tokens, 
                                weights=weights, smoothing_function=smoothing)
            results[n] = score
            
        return results
    except Exception as e:
        print(f"Lỗi khi tính BLEU-{n_values}: {e}")
        return {n: 0.0 for n in n_values}

# --- Các hàm gọi API và đánh giá cũ giữ nguyên (có thể copy từ evaluate_rag_system.py) ---
def call_rag_system(question: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            request_payload = {
                "query": question,
                "knowledgeBaseId": "3c1f6515-10d8-437b-9c51-c5351ddd2519",
                "config_id": "7489b23e-cc56-4d44-9f27-00da658d7fcc",
                "parameters": {"max_chunks": 3},
                "chat_section_id": None
            }
            print(f"🔄 Gọi RAG API (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(RAG_API_URL, json=request_payload, timeout=30)
            if response.status_code == 500:
                print(f"⚠️ Server Error 500 - attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    print("⏳ Đợi 5 giây trước khi thử lại...")
                    time.sleep(5)
                    continue
                else:
                    print("❌ Tất cả attempts đều thất bại với lỗi 500")
                    return "", []
            response.raise_for_status()
            rag_output = response.json()
            generated_answer = rag_output.get("llm_answer", "")
            retrieved_chunks_data = rag_output.get("retrieved_chunks", [])
            retrieved_contexts = []
            if isinstance(retrieved_chunks_data, list):
                for chunk_data in retrieved_chunks_data:
                    if isinstance(chunk_data, dict) and "chunk_text" in chunk_data:
                        retrieved_contexts.append(chunk_data["chunk_text"])
            print(f"✅ RAG API thành công - Answer length: {len(generated_answer)} chars, Contexts: {len(retrieved_contexts)}")
            return generated_answer, retrieved_contexts
        except Exception as e:
            print(f"❌ Lỗi khi gọi API RAG: {e}")
            if attempt < max_retries - 1:
                print("⏳ Đợi 3 giây trước khi thử lại...")
                time.sleep(3)
            else:
                print("❌ Tất cả attempts đều thất bại")
                return "", []
    return "", []

def call_gemini(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Lỗi Gemini attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "ERROR"

def evaluate_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    if not answer or not contexts:
        return 0.0
    context_text = "\n".join(contexts)
    prompt = f"""
Bạn là một chuyên gia đánh giá AI. Hãy đánh giá độ trung thực của câu trả lời dựa trên ngữ cảnh được cung cấp.
Câu hỏi: {question}
Ngữ cảnh:
{context_text}
Câu trả lời: {answer}
Hãy đánh giá từ 0-10 (10 là hoàn toàn trung thực, 0 là hoàn toàn không trung thực):
Chỉ trả lời bằng một số từ 0-10:
"""
    result = call_gemini(prompt)
    try:
        score = float(result)
        return min(max(score / 10.0, 0.0), 1.0)
    except:
        print(f"Không thể parse faithfulness score: {result}")
        return 0.5

def evaluate_relevancy(question: str, answer: str) -> float:
    if not answer:
        return 0.0
    prompt = f"""
Bạn là một chuyên gia đánh giá AI. Hãy đánh giá độ liên quan của câu trả lời với câu hỏi.
Câu hỏi: {question}
Câu trả lời: {answer}
Hãy đánh giá từ 0-10 (10 là hoàn toàn liên quan, 0 là hoàn toàn không liên quan):
Chỉ trả lời bằng một số từ 0-10:
"""
    result = call_gemini(prompt)
    try:
        score = float(result)
        return min(max(score / 10.0, 0.0), 1.0)
    except:
        print(f"Không thể parse relevancy score: {result}")
        return 0.5

def evaluate_context_recall(ground_truth: str, contexts: List[str]) -> float:
    if not ground_truth or not contexts:
        return 0.0
    context_text = "\n".join(contexts)
    prompt = f"""
Bạn là một chuyên gia đánh giá AI. Hãy đánh giá xem ngữ cảnh có chứa đủ thông tin để trả lời câu hỏi không.
Câu trả lời đúng (ground truth): {ground_truth}
Ngữ cảnh được truy xuất:
{context_text}
Hãy đánh giá từ 0-10 (10 là ngữ cảnh chứa đầy đủ thông tin, 0 là không chứa thông tin cần thiết):
Chỉ trả lời bằng một số từ 0-10:
"""
    result = call_gemini(prompt)
    try:
        score = float(result)
        return min(max(score / 10.0, 0.0), 1.0)
    except:
        print(f"Không thể parse context recall score: {result}")
        return 0.5

def calculate_similarity_score(answer: str, ground_truth: str) -> float:
    if not answer or not ground_truth:
        return 0.0
    
    import re
    
    # Chuẩn hóa text: chuyển về lowercase, strip, và loại bỏ dấu câu
    def clean_text(text):
        # Loại bỏ dấu câu và ký tự đặc biệt, chỉ giữ lại chữ cái và số
        cleaned = re.sub(r'[^\w\s]', '', text.lower().strip())
        return cleaned
    
    answer_cleaned = clean_text(answer)
    ground_truth_cleaned = clean_text(ground_truth)
    
    answer_words = set(answer_cleaned.split())
    ground_truth_words = set(ground_truth_cleaned.split())
    
    if len(ground_truth_words) == 0:
        return 0.0
    
    overlap = len(answer_words.intersection(ground_truth_words))
    similarity = overlap / len(ground_truth_words)
    
    return min(similarity, 1.0)

# --- Main pipeline ---
def main():
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"Không tìm thấy file: {EVAL_DATASET_PATH}")
        return
    print(f"🔄 Đang load dataset từ: {EVAL_DATASET_PATH}...")
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    print(f"✅ Đã load {len(qa_dataset)} Q&A pairs.")
    max_questions = 20
    if max_questions:
        qa_dataset = qa_dataset[:max_questions]
        print(f"🎯 Sẽ test với {max_questions} câu hỏi đầu tiên")
    else:
        print(f"🎯 Sẽ test với toàn bộ {len(qa_dataset)} câu hỏi trong dataset")
    total_questions = len(qa_dataset)
    print("\n🚀 Bắt đầu evaluation...")
    print("=" * 60)
    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S", time.localtime(start_time))
    print(f"🕐 Thời gian bắt đầu: {start_time_str}")
    results = []
    for i, item in enumerate(qa_dataset):
        question = item["question"]
        ground_truth = item["ground_truth_answer"]
        relevant_contexts = item.get("relevant_contexts", [])  # Nếu có ground truth context
        print(f"\n📝 Đang xử lý câu {i+1}/{total_questions}")
        print(f"❓ Câu hỏi: {question[:20]}...")
        generated_answer, retrieved_contexts = call_rag_system(question)
        if not generated_answer:
            print("❌ Không có câu trả lời từ RAG system")
            continue
        print(f"💬 Câu trả lời: {generated_answer[:20]}...")
        
        # Debug contexts removed for production
        faithfulness = evaluate_faithfulness(question, generated_answer, retrieved_contexts)
        relevancy = evaluate_relevancy(question, generated_answer)
        context_recall = evaluate_context_recall(ground_truth, retrieved_contexts)
        similarity = calculate_similarity_score(generated_answer, ground_truth)
        # --- Metrics mới ---
        k = 5
        prec_at_k = precision_at_k_semantic(retrieved_contexts, relevant_contexts, k)
        rec_at_k = recall_at_k_semantic(retrieved_contexts, relevant_contexts, k)
        mrr_score = mrr_semantic(retrieved_contexts, relevant_contexts)
        ndcg_score = ndcg_semantic(retrieved_contexts, relevant_contexts, k)
        
        # --- Metrics mới với Recall@K multiple values ---
        k_values = [1, 2, 3, 4, 5]
        recall_at_k_scores = recall_at_k_multiple_k(retrieved_contexts, relevant_contexts, k_values)
        
        # --- BLEU scores multiple n-grams ---
        bleu_scores = bleu_scores_multiple_n(ground_truth, generated_answer, [1, 4])
        
        # Debug text normalization cho BLEU và ROUGE
        debug_text_normalization(ground_truth, generated_answer)
        
        bleu = bleu_score(ground_truth, generated_answer)
        rouge = rouge_score(ground_truth, generated_answer)
        sem_sim = semantic_similarity(ground_truth, generated_answer)
        result = {
            'question_id': i + 1,
            'question': question,
            'generated_answer': generated_answer,
            'ground_truth': ground_truth,
            'retrieved_contexts_count': len(retrieved_contexts),
            'faithfulness_score': round(faithfulness, 3),
            'relevancy_score': round(relevancy, 3),
            'context_recall_score': round(context_recall, 3),
            'similarity_score': round(similarity, 3),
            'precision@k': round(prec_at_k, 3),
            'recall@k': round(rec_at_k, 3),
            # --- Thêm Recall@K cho nhiều giá trị ---
            'recall@1': round(recall_at_k_scores[1], 3),
            'recall@2': round(recall_at_k_scores[2], 3),
            'recall@3': round(recall_at_k_scores[3], 3),
            'recall@4': round(recall_at_k_scores[4], 3),
            'recall@5': round(recall_at_k_scores[5], 3),
            'mrr': round(mrr_score, 3),
            'ndcg': round(ndcg_score, 3),
            'bleu': round(bleu, 3),  # BLEU tổng hợp (giữ nguyên)
            # --- Thêm BLEU-1 và BLEU-4 ---
            'bleu-1': round(bleu_scores[1], 3),
            'bleu-4': round(bleu_scores[4], 3),
            'rouge-1': round(rouge['rouge-1']['f'], 3),
            'rouge-2': round(rouge['rouge-2']['f'], 3),
            'rouge-l': round(rouge['rouge-l']['f'], 3),
            'semantic_similarity': round(sem_sim, 3),
            'overall_score': round((faithfulness + relevancy + context_recall + similarity + 
                                  prec_at_k + rec_at_k + mrr_score + ndcg_score + 
                                  bleu + bleu_scores[1] + bleu_scores[4] + sem_sim) / 13, 3)  # Cập nhật overall score
        }
        results.append(result)
        print(f"📊 Scores - F:{faithfulness:.3f} R:{relevancy:.3f} CR:{context_recall:.3f} S:{similarity:.3f} P@K:{prec_at_k:.3f} R@K:{rec_at_k:.3f} MRR:{mrr_score:.3f} nDCG:{ndcg_score:.3f} BLEU:{bleu:.3f} R1:{rouge['rouge-1']['f']:.3f} R2:{rouge['rouge-2']['f']:.3f} RL:{rouge['rouge-l']['f']:.3f} SemSim:{sem_sim:.3f}")
        
        # Hiển thị chi tiết từng metric
        print(f"🔍 Chi tiết metrics:")
        print(f"  📊 Core Quality: F={faithfulness:.3f}, R={relevancy:.3f}, CR={context_recall:.3f}, S={similarity:.3f}")
        print(f"  🎯 Retrieval: P@K={prec_at_k:.3f}, R@K={rec_at_k:.3f}, R@1={recall_at_k_scores[1]:.3f}, R@2={recall_at_k_scores[2]:.3f}, R@3={recall_at_k_scores[3]:.3f}, R@4={recall_at_k_scores[4]:.3f}, R@5={recall_at_k_scores[5]:.3f}, MRR={mrr_score:.3f}, nDCG={ndcg_score:.3f}")
        print(f"  📝 Text Quality: BLEU={bleu:.3f}, BLEU-1={bleu_scores[1]:.3f}, BLEU-4={bleu_scores[4]:.3f}, R1={rouge['rouge-1']['f']:.3f}, R2={rouge['rouge-2']['f']:.3f}, RL={rouge['rouge-l']['f']:.3f}")
        print(f"  🧠 Semantic: SemSim={sem_sim:.3f}")
        print(f"  🏆 Overall Score: {result['overall_score']:.3f}")
        print(f"⏳ Đợi {DELAY_BETWEEN_QUESTIONS} giây trước câu hỏi tiếp theo...")
        time.sleep(DELAY_BETWEEN_QUESTIONS)
    # Lưu kết quả
    output_json = "rag_evaluation_results_extra_metrics_english_3.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Kết quả chi tiết đã lưu: {output_json}")
    df = pd.DataFrame(results)
    output_csv = "rag_evaluation_summary_extra_metrics.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"📊 Bảng tóm tắt đã lưu: {output_csv}")
    
    # Hiển thị bảng tóm tắt cuối cùng
    if results:
        print("\n" + "=" * 80)
        print("🎉 BẢNG TÓM TẮT TẤT CẢ METRICS ĐÃ ĐÁNH GIÁ")
        print("=" * 80)
        
        # Tính điểm trung bình cho tất cả metrics mới
        avg_scores = {}
        for key in ['faithfulness_score', 'relevancy_score', 'context_recall_score', 'similarity_score', 
                   'precision@k', 'recall@k', 'recall@1', 'recall@2', 'recall@3', 'recall@4', 'recall@5',
                   'mrr', 'ndcg', 'bleu', 'bleu-1', 'bleu-4', 'rouge-1', 'rouge-2', 'rouge-l', 
                   'semantic_similarity', 'overall_score']:
            values = [r.get(key, 0) for r in results if r.get(key) is not None]
            if values:
                avg_scores[key] = sum(values) / len(values)
        
        # Hiển thị theo nhóm mới
        print(f"{'📊 Core Quality':<20} {'📈 Retrieval':<25} {'📝 Text Quality':<25} {'🧠 Semantic':<15}")
        print("-" * 85)
        print(f"{'Faithfulness:':<15} {avg_scores.get('faithfulness_score', 0):.3f}    {'P@K:':<15} {avg_scores.get('precision@k', 0):.3f}    {'BLEU:':<15} {avg_scores.get('bleu', 0):.3f}    {'SemSim:':<15} {avg_scores.get('semantic_similarity', 0):.3f}")
        print(f"{'Relevancy:':<15} {avg_scores.get('relevancy_score', 0):.3f}    {'R@K:':<15} {avg_scores.get('recall@k', 0):.3f}    {'BLEU-1:':<15} {avg_scores.get('bleu-1', 0):.3f}    {'Overall:':<15} {avg_scores.get('overall_score', 0):.3f}")
        print(f"{'Context Recall:':<15} {avg_scores.get('context_recall_score', 0):.3f}    {'R@1:':<15} {avg_scores.get('recall@1', 0):.3f}    {'BLEU-4:':<15} {avg_scores.get('bleu-4', 0):.3f}")
        print(f"{'Similarity:':<15} {avg_scores.get('similarity_score', 0):.3f}    {'R@2:':<15} {avg_scores.get('recall@2', 0):.3f}    {'R1:':<15} {avg_scores.get('rouge-1', 0):.3f}")
        print(f"{'':<15} {'R@3:':<15} {avg_scores.get('recall@3', 0):.3f}    {'R2:':<15} {avg_scores.get('rouge-2', 0):.3f}")
        print(f"{'':<15} {'R@4:':<15} {avg_scores.get('recall@4', 0):.3f}    {'RL:':<15} {avg_scores.get('rouge-l', 0):.3f}")
        print(f"{'':<15} {'R@5:':<15} {avg_scores.get('recall@5', 0):.3f}")
        
        print("\n" + "=" * 80)
    
    print("\n✅ Hoàn thành evaluation!")

if __name__ == "__main__":
    main()