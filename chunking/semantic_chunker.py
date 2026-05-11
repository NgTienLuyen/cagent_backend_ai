# semantic_chunker.py
import re
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import logging
import pandas as pd
from services.keyword_extractor import extract_keywords

# Thêm import underthesea
try:
    from underthesea import word_tokenize, sent_tokenize as vi_sent_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    logging.warning("Thư viện underthesea không được tìm thấy. Xử lý tiếng Việt sẽ sử dụng NLTK.")

# Thiết lập logging
logger = logging.getLogger(__name__)

# Class định nghĩa cấu hình chunking
class ChunkingConfig(BaseModel):
    threshold: float = 0.3
    embedding_type: str = "sentence_transformers"
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    min_sentence_length: int = 4
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    overlap_size: int = 0
    clean_text: bool = True
    language: str = "vietnamese"
    use_clustering: bool = True  # Có sử dụng clustering hay không
    preserve_order: bool = True   # Có duy trì thứ tự ban đầu không
    verbose_logging: bool = False  # Ghi log chi tiết cho ma trận và kết quả phân cụm
    clustering_method: Literal["agglomerative", "hdbscan"] = "hdbscan"  # Phương pháp phân cụm
    min_cluster_size: int = 10  #Số lượng câu tối thiểu để tạo một cluster (cho HDBSCAN)
    outlier_handling: Literal["separate", "nearest", "singleton"] = "nearest"  # Cách xử lý outliers

    # Thêm cấu hình cho trích xuất từ khóa
    keyword_extraction_method: Literal["nlp", "llm"] = "nlp"
    keyword_extraction_llm_config: Optional[Dict[str, Any]] = None # Cấu hình LLM riêng cho keywords nếu cần

    # Các phương thức để lấy cấu hình mặc định và kiểm tra giá trị
    @classmethod
    def get_default_config(cls):
        return cls()

    @classmethod
    def get_available_models(cls):
        return {
            "sentence_transformers": [
                "all-mpnet-base-v2",
                "all-MiniLM-L6-v2",
                "paraphrase-multilingual-MiniLM-L12-v2",
                "paraphrase-multilingual-mpnet-base-v2",
                "distiluse-base-multilingual-cased-v1",
                "VietAI/viet-sbert-base"
            ],
            "tfidf": ["default"]
        }

    @classmethod
    def get_available_embedding_types(cls):
        return ["sentence_transformers", "tfidf"]

    @classmethod
    def get_available_languages(cls):
        return ["english", "vietnamese", "multi"]
    
    @classmethod
    def get_available_clustering_methods(cls):
        return ["agglomerative", "hdbscan"]
    
    @classmethod
    def get_available_outlier_handling_methods(cls):
        return ["separate", "nearest", "singleton"]


class SemanticChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None, llm_instance_for_keywords: Optional[Any] = None):
        # Sử dụng config mặc định nếu không có config được truyền vào
        self.config = config or ChunkingConfig()
        self.llm_instance_for_keywords = llm_instance_for_keywords
        
        logger.info(f"Khởi tạo SemanticChunker với cấu hình: {self.config.dict()}")
        logger.info(f"Phương pháp phân cụm ({self.config.clustering_method}): {'Kích hoạt' if self.config.use_clustering else 'Không sử dụng'}")

        nltk.download("punkt", quiet=True)

        if self.config.embedding_type in ["transformers", "sentence_transformers"]:
            logger.info(f"Đang tải model embedding: {self.config.model}")
            self.model = SentenceTransformer(self.config.model)
        elif self.config.embedding_type == "tfidf":
            self.model = None
        else:
            raise ValueError("Unsupported embedding type. Choose 'tfidf' or 'sentence_transformers'.")

    def embed_function(self, sentences):
        """Chuyển đổi câu thành vector embedding."""
        if self.config.embedding_type == "tfidf":
            vectorizer = TfidfVectorizer().fit_transform(sentences)
            return vectorizer.toarray()
        elif self.config.embedding_type in ["transformers", "sentence_transformers"]:
            return self.model.encode(sentences, convert_to_numpy=True)
        else:
            raise ValueError("Unsupported embedding type. Choose 'tfidf' or 'sentence_transformers'.")

    def clean_sentence(self, text):
        """Làm sạch văn bản nếu cần."""
        if not self.config.clean_text:
            return text

        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        # Giữ lại nhiều ký tự đặc biệt quan trọng hơn
        # Sửa lỗi SyntaxError bằng cách thoát dấu nháy đơn bên trong regex
        text = re.sub(r'[^\w\s\.,;?!\-\(\)\[\]\{\}:\"\'\/@#$%^&*+=_<>`~|]', '', text)
        
        # Sử dụng underthesea cho tiếng Việt nếu có
        if self.config.language == "vietnamese" and UNDERTHESEA_AVAILABLE:
            # Tokenize rồi join lại để chuẩn hóa từ ghép
            words = word_tokenize(text)
            text = " ".join([w.replace("_", " ") for w in words])
            
        return text.strip()

    async def _extract_keywords_for_chunk(self, chunk_text: str) -> List[str]:
        """Hàm helper để trích xuất từ khóa cho một chunk text."""
        # Quyết định llm_instance dựa trên việc nó có được truyền vào __init__ không
        # hoặc tải động nếu có keyword_extraction_llm_config
        llm_instance_to_use = self.llm_instance_for_keywords
        # if self.config.keyword_extraction_method == "llm" and not llm_instance_to_use and self.config.keyword_extraction_llm_config:
        #     try:
        #         logger.info(f"Loading LLM for keyword extraction using config: {self.config.keyword_extraction_llm_config.get('model_name')}")
        #         llm_instance_to_use = ModelConfigLoader.load_model(self.config.keyword_extraction_llm_config)
        #     except Exception as e:
        #         logger.error(f"Failed to load LLM for keyword extraction: {e}. Falling back to NLP.")
        #         return await extract_keywords(chunk_text, method="nlp") # Fallback

        return await extract_keywords(
            chunk_text,
            method=self.config.keyword_extraction_method,
            llm_instance=llm_instance_to_use
        )

    async def split_text_with_clustering(self, text: str) -> List[Dict[str, Any]]:
        """Phân đoạn văn bản sử dụng phương pháp phân cụm (clustering) và trích xuất keywords."""
        logger.info(f"Bắt đầu quá trình phân đoạn với phương pháp {self.config.clustering_method.upper()}")
        
        # Phân tách văn bản thành các câu - tích hợp underthesea cho tiếng Việt
        if self.config.language == "vietnamese" and UNDERTHESEA_AVAILABLE:
            logger.info("Sử dụng underthesea.sent_tokenize cho phân đoạn câu tiếng Việt")
            sentences = vi_sent_tokenize(text)
        else:
            sentences = nltk.sent_tokenize(text)
            
        original_sentences_count = len(sentences)
        sentences = [self.clean_sentence(sent) for sent in sentences 
                     if len(sent.strip()) > self.config.min_sentence_length]
        
        # NEW: Filter out marker-like sentences
        def is_marker_sentence(s, config):
            # Matches patterns like "1.", "1.2.", "1.2.3", "1.2.3.", " Điều 1.2.3."
            # Allows for common section prefixes.
            # Consider making prefixes configurable if needed
            pattern = r"^\s*(?:Điều|Chương|Mục|Phần|Phụ lục|Article|Section|Part|Appendix)?\s*(\d+\.)+(\d+)?\s*\.?\s*$"
            if config.verbose_logging and re.match(pattern, s, re.IGNORECASE):
                logger.info(f"Identified potential marker: '{s}'")
            return bool(re.match(pattern, s, re.IGNORECASE))

        sentences_before_marker_filter = len(sentences)
        sentences = [s for s in sentences if not is_marker_sentence(s, self.config)]
        if sentences_before_marker_filter > len(sentences):
             logger.info(f"Lọc bỏ {sentences_before_marker_filter - len(sentences)} câu giống như mục lục/marker.")
        
        logger.info(f"Tổng số câu sau khi lọc (từ {original_sentences_count} câu gốc, sau lọc độ dài và lọc marker): {len(sentences)}")
        
        if not sentences:
            logger.info("Không có câu nào đủ điều kiện sau khi lọc, trả về danh sách rỗng.")
            return []

        if len(sentences) <= 1 and not self.config.use_clustering:
            logger.info("Số câu ≤ 1 và không dùng clustering, trả về câu đó làm một chunk.")
            chunk_text = sentences[0]
            keywords = await self._extract_keywords_for_chunk(chunk_text)
            return [{"text": chunk_text, "keywords": keywords}]
            
        # Tạo embedding cho tất cả câu
        logger.info(f"Tạo embedding cho {len(sentences)} câu sử dụng {self.config.embedding_type}")
        embeddings = self.embed_function(sentences)
        
        # Tính toán ma trận tương đồng
        logger.info("Tính toán ma trận tương đồng (similarity matrix)")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Log ma trận tương đồng chi tiết
        if self.config.verbose_logging:
            # Giới hạn số lượng hiển thị nếu ma trận quá lớn
            max_display = min(10, len(sentences))
            logger.info(f"======= MA TRẬN TƯƠNG ĐỒNG (mẫu {max_display}x{max_display} đầu tiên) =======")
            
            # Tạo DataFrame với nhãn để dễ đọc
            sample_sentences = [s[:30] + "..." if len(s) > 30 else s for s in sentences[:max_display]]
            sample_matrix = similarity_matrix[:max_display, :max_display]
            
            df = pd.DataFrame(sample_matrix, 
                             index=sample_sentences,
                             columns=sample_sentences)
            
            # Log DataFrame dưới dạng chuỗi
            logger.info("\n" + df.to_string(float_format=lambda x: f"{x:.3f}"))
            
            # Thống kê phân bố giá trị tương đồng
            flat_sim = similarity_matrix.flatten()
            flat_sim = flat_sim[flat_sim < 0.999]  # Loại bỏ các giá trị tự so sánh (1.0)
            
            logger.info("Thống kê giá trị tương đồng:")
            logger.info(f"Min: {np.min(flat_sim):.3f}, Max: {np.max(flat_sim):.3f}, Mean: {np.mean(flat_sim):.3f}, Median: {np.median(flat_sim):.3f}")
            logger.info(f"Phân vị 25%: {np.percentile(flat_sim, 25):.3f}, 75%: {np.percentile(flat_sim, 75):.3f}")
        
        # Chọn phương pháp phân cụm và thực hiện phân cụm
        if self.config.clustering_method == "agglomerative":
            # Chuyển đổi ma trận tương đồng thành ma trận khoảng cách
            distance_matrix = 1 - similarity_matrix
            
            # Thực hiện phân cụm phân cấp
            logger.info(f"Thực hiện phân cụm phân cấp (hierarchical clustering) với ngưỡng {1 - self.config.threshold}")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.config.threshold,
                affinity='precomputed',
                linkage='average'
            ).fit(distance_matrix)
            
        else:  # hdbscan
            # Sử dụng trực tiếp min_cluster_size từ config
            effective_min_cluster_size = self.config.min_cluster_size

            # Sử dụng hdbscan_min_samples từ config nếu có, nếu không thì dùng giá trị mặc định nhỏ
            # (Giả sử hdbscan_min_samples đã được thêm vào ChunkingConfig và có giá trị mặc định)
            if hasattr(self.config, 'hdbscan_min_samples') and self.config.hdbscan_min_samples is not None and self.config.hdbscan_min_samples > 0:
                effective_min_samples = self.config.hdbscan_min_samples
            else:
                # Fallback nếu hdbscan_min_samples không có hoặc không hợp lệ
                effective_min_samples = max(1, int(effective_min_cluster_size * 0.5)) # Giữ lại logic cũ làm fallback HOẶC đặt một giá trị nhỏ cố định
                # Hoặc một giá trị mặc định nhỏ hơn, ví dụ:
                # effective_min_samples = 2 
                # Cân nhắc: nếu min_cluster_size rất nhỏ (ví dụ 1 hoặc 2), min_samples có thể cần được điều chỉnh để không lớn hơn min_cluster_size
                if effective_min_cluster_size < effective_min_samples:
                     effective_min_samples = effective_min_cluster_size # Đảm bảo min_samples không lớn hơn min_cluster_size
                if effective_min_samples <= 0: # Đảm bảo min_samples luôn dương
                    effective_min_samples = 1


            logger.info(f"Thực hiện phân cụm phân cấp dựa trên mật độ (HDBSCAN)")
            logger.info(f"Tham số: min_cluster_size={effective_min_cluster_size}, min_samples={effective_min_samples}, epsilon={1 - self.config.threshold}")
            
            # Đảm bảo các giá trị này không vượt quá số lượng câu
            num_sentences = len(sentences)
            if num_sentences == 0: # Xử lý trường hợp không có câu nào
                 logger.warning("Không có câu nào để phân cụm với HDBSCAN sau khi lọc.")
                 clusters = np.array([])
            elif num_sentences < effective_min_cluster_size :
                 logger.warning(f"Số lượng câu ({num_sentences}) ít hơn min_cluster_size ({effective_min_cluster_size}). Tất cả sẽ được coi là outliers.")
                 clusters = np.full(num_sentences, -1, dtype=int) # Tất cả là outliers
            elif num_sentences < effective_min_samples:
                 logger.warning(f"Số lượng câu ({num_sentences}) ít hơn min_samples ({effective_min_samples}). Điều chỉnh min_samples = {num_sentences}.")
                 effective_min_samples = num_sentences # hoặc xử lý như outliers
                 clustering = hdbscan.HDBSCAN(
                    min_cluster_size=effective_min_cluster_size,
                    min_samples=effective_min_samples, # đã điều chỉnh
                    metric='euclidean',
                    cluster_selection_epsilon=1 - self.config.threshold,
                    gen_min_span_tree=True
                 ).fit(embeddings)
                 clusters = clustering.labels_.copy()
            else:
                clustering = hdbscan.HDBSCAN(
                    min_cluster_size=effective_min_cluster_size,
                    min_samples=effective_min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=1 - self.config.threshold,
                    gen_min_span_tree=True
                ).fit(embeddings)
                clusters = clustering.labels_.copy()
        
        # Lấy nhãn cluster cho mỗi câu
        # clusters = clustering.labels_.copy() # Đã gán ở trên rồi
        
        num_clusters_before = len(set(clusters) - {-1} if -1 in clusters else set(clusters))
        logger.info(f"Kết quả phân cụm ban đầu: {num_clusters_before} clusters từ {len(sentences)} câu")
        
        # Xử lý outliers cho HDBSCAN (nhãn -1)
        if self.config.clustering_method == "hdbscan" and -1 in clusters:
            num_outliers = np.sum(clusters == -1)
            logger.info(f"Phát hiện {num_outliers} câu outliers (nhãn -1) với HDBSCAN")
            
            if self.config.outlier_handling == "separate":
                # Đánh số mỗi outlier là một cluster riêng
                next_cluster_id = clusters.max() + 1
                for idx in range(len(clusters)):
                    if clusters[idx] == -1:
                        clusters[idx] = next_cluster_id
                        next_cluster_id += 1
                logger.info(f"Xử lý outliers: Tạo cluster riêng cho mỗi outlier ({num_outliers} clusters mới)")
                
            elif self.config.outlier_handling == "singleton":
                # Tạo một cluster chung cho tất cả outliers
                next_cluster_id = clusters.max() + 1
                for idx in range(len(clusters)):
                    if clusters[idx] == -1:
                        clusters[idx] = next_cluster_id
                logger.info(f"Xử lý outliers: Tạo 1 cluster chung cho tất cả outliers")
                
            else:  # "nearest"
                # Gán outlier vào cluster gần nhất
                non_outlier_indices = np.where(clusters != -1)[0]
                outlier_indices = np.where(clusters == -1)[0]
                
                if len(non_outlier_indices) > 0:
                    for idx in outlier_indices:
                        similarities_to_clusters = similarity_matrix[idx, non_outlier_indices]
                        nearest_cluster_idx = non_outlier_indices[np.argmax(similarities_to_clusters)]
                        clusters[idx] = clusters[nearest_cluster_idx]
                    logger.info(f"Xử lý outliers: Gán mỗi outlier vào cluster gần nhất")
                else:
                    # Nếu tất cả đều là outlier, tạo một cluster duy nhất
                    clusters[:] = 0
                    logger.info(f"Xử lý outliers: Tất cả đều là outlier, tạo 1 cluster duy nhất")
        
        unique_clusters = sorted(set(clusters))
        logger.info(f"Kết quả phân cụm cuối cùng: {len(unique_clusters)} clusters từ {len(sentences)} câu")
        
        # Log chi tiết về kết quả phân cụm
        if self.config.verbose_logging:
            logger.info("======= KẾT QUẢ PHÂN CỤM CHI TIẾT =======")
            
            # Thống kê số lượng câu trong mỗi cluster
            cluster_stats = {}
            for label in clusters:
                if label not in cluster_stats:
                    cluster_stats[label] = 0
                cluster_stats[label] += 1
            
            for cluster_id in sorted(cluster_stats.keys()):
                logger.info(f"Cluster {cluster_id}: {cluster_stats[cluster_id]} câu")
            
            # Hiển thị một số mẫu từ mỗi cluster
            samples_per_cluster = 2  # Số câu mẫu hiển thị cho mỗi cluster
            
            for cluster_id in unique_clusters:
                sample_indices = [i for i, label in enumerate(clusters) if label == cluster_id][:samples_per_cluster]
                sample_texts = [sentences[i] for i in sample_indices]
                
                logger.info(f"Mẫu câu từ Cluster {cluster_id}:")
                for i, text in enumerate(sample_texts):
                    if len(text) > 100:
                        text = text[:97] + "..."
                    logger.info(f"  - Mẫu {i+1}: {text}")
            
            # Hiển thị ma trận phân cụm - phân bố nhãn cluster cho N câu đầu tiên
            max_display_sentences = min(20, len(sentences))
            sample_data = []
            
            for i in range(max_display_sentences):
                short_sent = sentences[i][:30] + "..." if len(sentences[i]) > 30 else sentences[i]
                sample_data.append([i, short_sent, clusters[i]])
                
            sample_df = pd.DataFrame(sample_data, columns=["Index", "Sentence", "Cluster"])
            logger.info(f"Phân bố nhãn cluster cho {max_display_sentences} câu đầu tiên:")
            logger.info("\n" + sample_df.to_string())
        
        # Thống kê các cluster
        cluster_stats = {}
        for label in clusters:
            if label not in cluster_stats:
                cluster_stats[label] = 0
            cluster_stats[label] += 1
            
        logger.info(f"Chi tiết phân cụm: {cluster_stats}")
        
        # Xử lý các cluster
        if self.config.preserve_order:
            logger.info("Bảo toàn thứ tự câu gốc trong quá trình gom nhóm")
            # Duy trì thứ tự xuất hiện
            sorted_indices = range(len(sentences))  # Chỉ mục nguyên bản
        else:
            logger.info("Sắp xếp lại thứ tự câu theo cluster")
            # Sắp xếp lại các câu theo cluster
            sorted_indices = sorted(range(len(sentences)), key=lambda i: clusters[i])
            
        # Gom nhóm các câu theo cluster
        result_chunks = []
        current_chunk = []
        current_cluster = clusters[sorted_indices[0]]
        current_length = 0
        
        for idx in sorted_indices:
            sentence = sentences[idx]
            cluster_id = clusters[idx]
            
            # Kiểm tra nếu là cluster mới hoặc chunk vượt quá kích thước
            if (cluster_id != current_cluster or 
                current_length + len(sentence) > self.config.max_chunk_size) and current_chunk:
                # Lưu chunk hiện tại
                chunk_text = " ".join(current_chunk)
                result_chunks.append(chunk_text)
                logger.info(f"Tạo chunk mới (cluster: {current_cluster}, độ dài: {current_length})")
                # Bắt đầu chunk mới
                current_chunk = [sentence]
                current_cluster = cluster_id
                current_length = len(sentence)
            else:
                # Tiếp tục thêm vào chunk hiện tại
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        # Thêm chunk cuối cùng
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            result_chunks.append(chunk_text)
            logger.info(f"Thêm chunk cuối cùng (cluster: {current_cluster}, độ dài: {current_length})")
            
        logger.info(f"Kết quả ban đầu: {len(result_chunks)} chunks")
            
        # Xử lý chunks quá nhỏ
        final_chunks = []
        merged_count = 0
        
        if self.config.verbose_logging:
            logger.info("====== CHI TIẾT CÁC CHUNKS BAN ĐẦU ======")
            for i, chunk in enumerate(result_chunks):
                preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                logger.info(f"Chunk {i+1} ({len(chunk)} ký tự): {preview}")
        
        for i, chunk in enumerate(result_chunks):
            if len(chunk) < self.config.min_chunk_size and i > 0:
                # Ghi log trước khi gộp nếu verbose_logging được bật
                if self.config.verbose_logging:
                    logger.info(f"\n====== GỘP CHUNK ======")
                    small_chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    previous_chunk_preview = final_chunks[-1][:100] + "..." if len(final_chunks[-1]) > 100 else final_chunks[-1]
                    logger.info(f"Chunk nhỏ cần gộp ({i+1}, {len(chunk)} ký tự): {small_chunk_preview}")
                    logger.info(f"Chunk trước đó ({i}, {len(final_chunks[-1])} ký tự): {previous_chunk_preview}")
                
                # Gộp với chunk trước đó
                final_chunks[-1] = final_chunks[-1] + " " + chunk
                merged_count += 1
                
                # Ghi log sau khi gộp nếu verbose_logging được bật
                if self.config.verbose_logging:
                    after_merge_preview = final_chunks[-1][:100] + "..." if len(final_chunks[-1]) > 100 else final_chunks[-1]
                    logger.info(f"Kết quả sau khi gộp ({len(final_chunks[-1])} ký tự): {after_merge_preview}")
            else:
                final_chunks.append(chunk)
                
        logger.info(f"Sau khi gộp chunks nhỏ: {len(final_chunks)} chunks (gộp {merged_count} chunks)")

        # Hiển thị kết quả cuối cùng nếu verbose_logging được bật
        if self.config.verbose_logging:
            logger.info("====== CHUNKS SAU KHI GỘP ======")
            for i, chunk in enumerate(final_chunks):
                preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                logger.info(f"Chunk {i+1} cuối cùng ({len(chunk)} ký tự): {preview}")
                
        # Xử lý overlap nếu cần
        if self.config.overlap_size > 0 and len(final_chunks) > 1:
            logger.info(f"Thêm overlap với kích thước {self.config.overlap_size} từ")
            overlapped_chunks_texts = []
            for i in range(len(final_chunks)):
                current_chunk_text = final_chunks[i]
                if i < len(final_chunks) - 1:
                    next_chunk_text = final_chunks[i + 1]
                    
                    if self.config.language == "vietnamese" and UNDERTHESEA_AVAILABLE:
                        words = word_tokenize(current_chunk_text)
                    else:
                        words = current_chunk_text.split()
                        
                    if len(words) > self.config.overlap_size:
                        if self.config.language == "vietnamese" and UNDERTHESEA_AVAILABLE:
                            overlap_text = ' '.join([w.replace("_", " ") for w in words[-self.config.overlap_size:]])
                        else:
                            overlap_text = ' '.join(words[-self.config.overlap_size:])
                        final_chunks[i + 1] = overlap_text + ' ' + next_chunk_text
                overlapped_chunks_texts.append(final_chunks[i])
            logger.info(f"Hoàn thành xử lý overlap với {len(overlapped_chunks_texts)} chunks text")
            # Gán lại final_chunks để trích xuất từ khóa trên phiên bản đã có overlap
            final_chunks = overlapped_chunks_texts 
            
        logger.info(f"Hoàn thành quá trình tạo {len(final_chunks)} chunks text.")

        # Trích xuất từ khóa cho mỗi chunk cuối cùng
        output_chunks_with_keywords = []
        for chunk_item_text in final_chunks:
            keywords = await self._extract_keywords_for_chunk(chunk_item_text)
            output_chunks_with_keywords.append({"text": chunk_item_text, "keywords": keywords})
            if self.config.verbose_logging:
                logger.info(f"Chunk: {chunk_item_text[:80]}... - Keywords: {keywords[:5]}")
        
        logger.info(f"Hoàn thành quá trình clustering và trích xuất từ khóa với {len(output_chunks_with_keywords)} chunks.")
        return output_chunks_with_keywords

    async def split_text(self, text: str) -> List[Dict[str, Any]]:
        """Phân đoạn văn bản thành các phần có ý nghĩa gần nhau và trích xuất keywords."""
        if not text or not text.strip():
            logger.info("Văn bản đầu vào rỗng hoặc chỉ chứa khoảng trắng.")
            return []
            
        # Làm sạch văn bản
        # text = self.clean_sentence(text) # clean_sentence đã được gọi trong các hàm con nếu cần
        
        # Kiểm tra nếu văn bản quá ngắn (sau khi có thể đã clean ở mức câu)
        # Quyết định này nên dựa vào logic sau khi tách câu, không phải ở đây ngay
        # if len(text) < self.config.min_chunk_size:
        #     keywords = await self._extract_keywords_for_chunk(text)
        #     return [{"text": text, "keywords": keywords}]
            
        # Sử dụng phương pháp clustering nếu được yêu cầu
        if self.config.use_clustering:
            logger.info(f"Phương pháp {self.config.clustering_method.upper()} được kích hoạt cho chunking.")
            return await self.split_text_with_clustering(text) # Đã async
            
        # Phương pháp truyền thống - so sánh tuần tự
        logger.info("Sử dụng phương pháp TRUYỀN THỐNG (so sánh tuần tự) cho chunking.")
        
        # Phân đoạn văn bản thành câu - tích hợp underthesea cho tiếng Việt
        if self.config.language == "vietnamese" and UNDERTHESEA_AVAILABLE:
            logger.info("Sử dụng underthesea.sent_tokenize cho phân đoạn câu tiếng Việt")
            sentences = vi_sent_tokenize(text)
        else:
            sentences = nltk.sent_tokenize(text)  # Phân đoạn văn bản thành câu
            
        sentences = [sent.strip() for sent in sentences if sent.strip()]  # Xóa câu trống

        # Làm sạch và lọc câu ngắn
        sentences = [
            self.clean_sentence(sent)
            for sent in sentences
            if len(sent.strip()) > self.config.min_sentence_length
        ]

        # NEW: Filter out marker-like sentences (also in traditional split)
        def is_marker_sentence(s, config): # Re-define or make it a helper method of the class
            pattern = r"^\s*(?:Điều|Chương|Mục|Phần|Phụ lục|Article|Section|Part|Appendix)?\s*(\d+\.)+(\d+)?\s*\.?\s*$"
            if config.verbose_logging and re.match(pattern, s, re.IGNORECASE):
                logger.info(f"Identified potential marker (traditional): '{s}'")
            return bool(re.match(pattern, s, re.IGNORECASE))

        sentences_before_marker_filter_trad = len(sentences)
        sentences = [s for s in sentences if not is_marker_sentence(s, self.config)]
        if sentences_before_marker_filter_trad > len(sentences):
            logger.info(f"Lọc bỏ {sentences_before_marker_filter_trad - len(sentences)} câu giống như mục lục/marker (phương pháp truyền thống).")

        if not sentences:
            logger.info("Không có câu nào sau khi tiền xử lý (phương pháp truyền thống).")
            return []

        if len(sentences) == 1:
            logger.info("Chỉ có một câu, trả về câu đó làm một chunk (phương pháp truyền thống).")
            keywords = await self._extract_keywords_for_chunk(sentences[0])
            return [{"text": sentences[0], "keywords": keywords}]

        vectors = self.embed_function(sentences)
        similarities = cosine_similarity(vectors)

        chunks = [[sentences[0]]]

        for i in range(1, len(sentences)):
            sim_score = similarities[i - 1, i] if i < len(similarities) else 0

            if sim_score >= self.config.threshold:
                chunks[-1].append(sentences[i])
            else:
                chunks.append([sentences[i]])

        # Xử lý kích thước chunk
        result_chunks_text = []
        for chunk in chunks:
            chunk_text = ' '.join(chunk)

            # Nếu chunk quá lớn, chia nhỏ hơn
            if len(chunk_text) > self.config.max_chunk_size and len(chunk) > 1:
                current_size = 0
                current_chunk = []

                for sentence in chunk:
                    if current_size + len(sentence) > self.config.max_chunk_size and current_chunk:
                        result_chunks_text.append(' '.join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    current_chunk.append(sentence)
                    current_size += len(sentence)

                if current_chunk:
                    result_chunks_text.append(' '.join(current_chunk))
            else:
                # Nếu chunk quá nhỏ, có thể gộp với chunk kế tiếp
                if len(chunk_text) < self.config.min_chunk_size and result_chunks_text:
                    result_chunks_text[-1] = result_chunks_text[-1] + ' ' + chunk_text
                else:
                    result_chunks_text.append(chunk_text) # Đổi tên biến để rõ ràng là list of strings

        # Trích xuất từ khóa cho mỗi chunk cuối cùng (phương pháp truyền thống)
        output_chunks_with_keywords_traditional = []
        for chunk_item_text in result_chunks_text: # Đổi tên biến
            keywords = await self._extract_keywords_for_chunk(chunk_item_text)
            output_chunks_with_keywords_traditional.append({"text": chunk_item_text, "keywords": keywords})
            if self.config.verbose_logging:
                 logger.info(f"Traditional Chunk: {chunk_item_text[:80]}... - Keywords: {keywords[:5]}")

        logger.info(f"Hoàn thành (phương pháp truyền thống) với {len(output_chunks_with_keywords_traditional)} chunks.")
        return output_chunks_with_keywords_traditional
