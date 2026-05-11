import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json

class TransformationMetricsCalculator:
    def __init__(self):
        self.results = {}
        
    def linear_projection_orthogonal(self, source_emb: np.ndarray, target_dim: int) -> np.ndarray:
        """Linear projection with orthogonal mapping"""
        source_dim = source_emb.shape[1]
        
        # Create orthogonal matrix using QR decomposition
        W = np.random.randn(target_dim, source_dim)
        Q, R = np.linalg.qr(W)
        W_orthogonal = Q[:, :source_dim]
        
        # Apply transformation
        transformed = W_orthogonal @ source_emb.T
        return transformed.T
    
    def dct_upsampling(self, source_emb: np.ndarray, target_dim: int) -> np.ndarray:
        """DCT upsampling method"""
        source_dim = source_emb.shape[1]
        
        # Apply DCT to each embedding
        dct_coeffs = dct(source_emb, axis=1)
        
        # Upsample by zero-padding
        upsampled_coeffs = np.zeros((source_emb.shape[0], target_dim))
        upsampled_coeffs[:, :source_dim] = dct_coeffs
        
        # Apply inverse DCT
        transformed = idct(upsampled_coeffs, axis=1)
        return transformed
    
    def weighted_redistribution(self, source_emb: np.ndarray, target_dim: int) -> np.ndarray:
        """Weighted redistribution method"""
        source_dim = source_emb.shape[1]
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(source_emb)
        
        # Create target embeddings
        transformed = np.zeros((source_emb.shape[0], target_dim))
        
        for i in range(source_emb.shape[0]):
            # Calculate weights based on similarity
            weights = np.exp(sim_matrix[i] / 0.1)  # temperature = 0.1
            weights = weights / np.sum(weights)
            
            # Redistribute information
            for j in range(target_dim):
                if j < source_dim:
                    transformed[i, j] = source_emb[i, j]
                else:
                    # Weighted combination of source dimensions
                    transformed[i, j] = np.sum(weights * source_emb[i, :])
        
        return transformed
    
    def pca_reduction(self, source_emb: np.ndarray, target_dim: int) -> np.ndarray:
        """PCA-based dimension reduction"""
        pca = PCA(n_components=target_dim)
        transformed = pca.fit_transform(source_emb)
        return transformed
    
    def calculate_semantic_preservation(self, original: np.ndarray, transformed: np.ndarray) -> float:
        """Calculate semantic preservation rate"""
        # Normalize embeddings
        original_norm = original / np.linalg.norm(original, axis=1, keepdims=True)
        transformed_norm = transformed / np.linalg.norm(transformed, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.diag(cosine_similarity(original_norm, transformed_norm))
        return np.mean(similarities)
    
    def calculate_cross_model_success(self, original: np.ndarray, transformed: np.ndarray, threshold: float = 0.8) -> float:
        """Calculate cross-model success rate"""
        similarities = self.calculate_semantic_preservation(original, transformed)
        success_rate = np.mean(similarities > threshold)
        return success_rate
    
    def measure_computational_cost(self, func, *args) -> Tuple[float, float]:
        """Measure computational time and memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - start_memory
        
        return end_time - start_time, memory_used
    
    def adaptive_strategy_selection(self, source_dim: int, target_dim: int, 
                                  computational_budget: float = 1.0, 
                                  semantic_requirement: float = 0.8) -> str:
        """Adaptive strategy selection algorithm"""
        dimension_ratio = target_dim / source_dim
        
        # Define weights
        dim_weight = 0.5
        budget_weight = 0.3
        semantic_weight = 0.2
        
        # Calculate scores for each method
        scores = {}
        
        # Linear projection score
        if dimension_ratio <= 2:
            scores['linear'] = dim_weight * 0.9 + budget_weight * 0.9 + semantic_weight * semantic_requirement
        else:
            scores['linear'] = dim_weight * 0.3 + budget_weight * 0.9 + semantic_weight * semantic_requirement
        
        # DCT upsampling score
        if 2 < dimension_ratio <= 4:
            scores['dct'] = dim_weight * 0.8 + budget_weight * 0.6 + semantic_weight * 0.9
        else:
            scores['dct'] = dim_weight * 0.4 + budget_weight * 0.6 + semantic_weight * 0.9
        
        # Weighted redistribution score
        if dimension_ratio > 4:
            scores['weighted'] = dim_weight * 0.9 + budget_weight * 0.7 + semantic_weight * 0.95
        else:
            scores['weighted'] = dim_weight * 0.6 + budget_weight * 0.7 + semantic_weight * 0.95
        
        # PCA reduction score
        if dimension_ratio < 0.5:
            scores['pca'] = dim_weight * 0.9 + budget_weight * 0.9 + semantic_weight * 0.7
        else:
            scores['pca'] = dim_weight * 0.2 + budget_weight * 0.9 + semantic_weight * 0.7
        
        # Return best method
        return max(scores, key=scores.get)
    
    def evaluate_transformation_methods(self, num_samples: int = 1000, 
                                     source_dim: int = 384, 
                                     target_dims: List[int] = None) -> Dict:
        """Comprehensive evaluation of all transformation methods"""
        if target_dims is None:
            target_dims = [192, 256, 512, 768, 1024]  # Various target dimensions
        
        # Generate synthetic embeddings
        np.random.seed(42)
        source_embeddings = np.random.randn(num_samples, source_dim)
        
        results = {
            'linear_projection': {},
            'dct_upsampling': {},
            'weighted_redistribution': {},
            'pca_reduction': {},
            'adaptive_selection': {}
        }
        
        # Test each method with different target dimensions
        for target_dim in target_dims:
            dimension_ratio = target_dim / source_dim
            
            print(f"Testing dimension ratio: {dimension_ratio:.2f} ({source_dim} -> {target_dim})")
            
            # Linear projection
            if dimension_ratio >= 1:
                start_time = time.time()
                linear_transformed = self.linear_projection_orthogonal(source_embeddings, target_dim)
                linear_time = time.time() - start_time
                
                linear_preservation = self.calculate_semantic_preservation(source_embeddings, linear_transformed)
                linear_success = self.calculate_cross_model_success(source_embeddings, linear_transformed)
                
                results['linear_projection'][dimension_ratio] = {
                    'preservation': linear_preservation,
                    'success_rate': linear_success,
                    'time': linear_time
                }
            
            # DCT upsampling
            if dimension_ratio >= 1:
                start_time = time.time()
                dct_transformed = self.dct_upsampling(source_embeddings, target_dim)
                dct_time = time.time() - start_time
                
                dct_preservation = self.calculate_semantic_preservation(source_embeddings, dct_transformed)
                dct_success = self.calculate_cross_model_success(source_embeddings, dct_transformed)
                
                results['dct_upsampling'][dimension_ratio] = {
                    'preservation': dct_preservation,
                    'success_rate': dct_success,
                    'time': dct_time
                }
            
            # Weighted redistribution
            if dimension_ratio >= 1:
                start_time = time.time()
                weighted_transformed = self.weighted_redistribution(source_embeddings, target_dim)
                weighted_time = time.time() - start_time
                
                weighted_preservation = self.calculate_semantic_preservation(source_embeddings, weighted_transformed)
                weighted_success = self.calculate_cross_model_success(source_embeddings, weighted_transformed)
                
                results['weighted_redistribution'][dimension_ratio] = {
                    'preservation': weighted_preservation,
                    'success_rate': weighted_success,
                    'time': weighted_time
                }
            
            # PCA reduction
            if dimension_ratio < 1:
                start_time = time.time()
                pca_transformed = self.pca_reduction(source_embeddings, target_dim)
                pca_time = time.time() - start_time
                
                pca_preservation = self.calculate_semantic_preservation(source_embeddings, pca_transformed)
                pca_success = self.calculate_cross_model_success(source_embeddings, pca_transformed)
                
                results['pca_reduction'][dimension_ratio] = {
                    'preservation': pca_preservation,
                    'success_rate': pca_success,
                    'time': pca_time
                }
            
            # Test adaptive selection
            start_time = time.time()
            selected_method = self.adaptive_strategy_selection(source_dim, target_dim)
            selection_time = time.time() - start_time
            
            results['adaptive_selection'][dimension_ratio] = {
                'selected_method': selected_method,
                'decision_time': selection_time
            }
        
        return results
    
    def generate_practical_guidelines(self, results: Dict) -> Dict:
        """Generate practical guidelines based on evaluation results"""
        guidelines = {
            'optimal_ranges': {},
            'performance_metrics': {},
            'adaptive_algorithm_stats': {}
        }
        
        # Analyze optimal ranges
        for method, data in results.items():
            if method == 'adaptive_selection':
                continue
                
            best_ratio = None
            best_preservation = 0
            
            for ratio, metrics in data.items():
                if metrics['preservation'] > best_preservation:
                    best_preservation = metrics['preservation']
                    best_ratio = ratio
            
            guidelines['optimal_ranges'][method] = {
                'best_ratio': best_ratio,
                'best_preservation': best_preservation
            }
        
        # Calculate adaptive algorithm statistics
        decision_times = []
        method_selections = {}
        
        for ratio, data in results['adaptive_selection'].items():
            decision_times.append(data['decision_time'])
            method = data['selected_method']
            method_selections[method] = method_selections.get(method, 0) + 1
        
        guidelines['adaptive_algorithm_stats'] = {
            'avg_decision_time': np.mean(decision_times),
            'method_distribution': method_selections,
            'total_decisions': len(decision_times)
        }
        
        return guidelines
    
    def create_visualizations(self, results: Dict, guidelines: Dict):
        """Create visualizations for the results"""
        # Performance comparison chart
        methods = ['linear_projection', 'dct_upsampling', 'weighted_redistribution', 'pca_reduction']
        ratios = []
        preservations = []
        method_names = []
        
        for method in methods:
            if method in results:
                for ratio, data in results[method].items():
                    ratios.append(ratio)
                    preservations.append(data['preservation'])
                    method_names.append(method.replace('_', ' ').title())
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Dimension Ratio': ratios,
            'Semantic Preservation': preservations,
            'Method': method_names
        })
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='Dimension Ratio', y='Semantic Preservation', 
                       hue='Method', s=100, alpha=0.7)
        plt.title('Semantic Preservation vs Dimension Ratio by Transformation Method')
        plt.xlabel('Dimension Ratio (d2/d1)')
        plt.ylabel('Semantic Preservation Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('transformation_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Adaptive selection performance
        decision_times = [data['decision_time'] for data in results['adaptive_selection'].values()]
        plt.figure(figsize=(8, 6))
        plt.hist(decision_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Adaptive Strategy Selection Decision Times')
        plt.xlabel('Decision Time (seconds)')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(decision_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(decision_times):.4f}s')
        plt.legend()
        plt.tight_layout()
        plt.savefig('adaptive_selection_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the comprehensive evaluation"""
    calculator = TransformationMetricsCalculator()
    
    print("Starting comprehensive evaluation of embedding transformation methods...")
    
    # Run evaluation
    results = calculator.evaluate_transformation_methods(
        num_samples=1000,
        source_dim=384,
        target_dims=[192, 256, 384, 512, 768, 1024]
    )
    
    # Generate guidelines
    guidelines = calculator.generate_practical_guidelines(results)
    
    # Create visualizations
    calculator.create_visualizations(results, guidelines)
    
    # Print results
    print("\n=== PRACTICAL GUIDELINES ===")
    print("Based on our comprehensive analysis, we provide practical guidelines for system designers:")
    
    optimal_ranges = guidelines['optimal_ranges']
    for method, data in optimal_ranges.items():
        method_name = method.replace('_', ' ').title()
        print(f"- {method_name}: Best performance at ratio {data['best_ratio']:.2f} "
              f"(preservation: {data['best_preservation']:.3f})")
    
    print("\n=== ADAPTIVE ALGORITHM STATISTICS ===")
    adaptive_stats = guidelines['adaptive_algorithm_stats']
    print(f"- Average decision time: {adaptive_stats['avg_decision_time']*1000:.1f}ms")
    print(f"- Method selection distribution: {adaptive_stats['method_distribution']}")
    
    # Save results
    with open('transformation_evaluation_results.json', 'w') as f:
        json.dump({
            'results': results,
            'guidelines': guidelines
        }, f, indent=2, default=str)
    
    print("\nResults saved to 'transformation_evaluation_results.json'")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main() 