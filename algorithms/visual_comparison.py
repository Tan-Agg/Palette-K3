import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import load_image_as_array
from naive_kmeans import NaiveKMeans
from kmeans_plusplus import KMeansPlusPlus
from naive_kmeans_lab import NaiveKMeansLAB
from lab_kmeans import LABKMeansPlusPlus

K = 3

def run_algorithm_multiple_times(AlgorithmClass, pixels, k, n_runs=20, is_lab=False):
    """Run algorithm multiple times, return best result"""
    best_result = None
    best_inertia = float('inf')
    
    print(f"  Running {n_runs} times to find best result...", end=' ')
    
    for seed in range(n_runs):
        algo = AlgorithmClass(k=k, random_state=seed, max_iter=100)
        algo.fit(pixels)
        
        # Use native inertia (what the algorithm optimizes for)
        native_inertia = algo.inertia
        
        if native_inertia < best_inertia:
            best_inertia = native_inertia
            best_result = algo
    
    print(f"Done! Best inertia: {best_inertia:.2f}")
    return best_result

def reconstruct_image(pixels, labels, centroids, original_shape):
    """Reconstruct image from clustered pixels"""
    reconstructed = centroids[labels]
    reconstructed_image = reconstructed.reshape(original_shape)
    return reconstructed_image.astype(np.uint8)

def run_visual_comparison(image_path, k=20, n_runs=20, save_path='visual_comparison.png'):
    """Run all 4 algorithms with multiple runs and create side-by-side comparison"""
    
    # Load image
    pixels, image_array, original_shape = load_image_as_array(image_path)
    print(f"Image loaded: {pixels.shape[0]} pixels, shape: {original_shape}")
    print(f"Running with k={k} colors, {n_runs} runs per algorithm\n")
    
    # Run all algorithms MULTIPLE TIMES
    algorithms = {}
    
    print("1. Naive K-Means (RGB)...")
    naive_rgb = run_algorithm_multiple_times(NaiveKMeans, pixels, k, n_runs, is_lab=False)
    algorithms['Naive K-Means\n(RGB)'] = {
        'labels': naive_rgb.labels,
        'centroids': naive_rgb.centroids,
        'iterations': naive_rgb.n_iterations,
        'inertia': naive_rgb.inertia,  # Only RGB
        'space': 'RGB'
    }
    
    print("2. K-Means++ (RGB)...")
    kpp_rgb = run_algorithm_multiple_times(KMeansPlusPlus, pixels, k, n_runs, is_lab=False)
    algorithms['K-Means++\n(RGB)'] = {
        'labels': kpp_rgb.labels,
        'centroids': kpp_rgb.centroids,
        'iterations': kpp_rgb.n_iterations,
        'inertia': kpp_rgb.inertia,  # Only RGB
        'space': 'RGB'
    }
    
    print("3. Naive K-Means (LAB)...")
    naive_lab = run_algorithm_multiple_times(NaiveKMeansLAB, pixels, k, n_runs, is_lab=True)
    algorithms['Naive K-Means\n(LAB)'] = {
        'labels': naive_lab.labels,
        'centroids': naive_lab.centroids_rgb,
        'iterations': naive_lab.n_iterations,
        'inertia': naive_lab.inertia,  # Only LAB
        'space': 'LAB'
    }
    
    print("4. K-Means++ (LAB)...")
    kpp_lab = run_algorithm_multiple_times(LABKMeansPlusPlus, pixels, k, n_runs, is_lab=True)
    algorithms['K-Means++\n(LAB)'] = {
        'labels': kpp_lab.labels,
        'centroids': kpp_lab.centroids_rgb,
        'iterations': kpp_lab.n_iterations,
        'inertia': kpp_lab.inertia,  # Only LAB
        'space': 'LAB'
    }
    
    print("\nReconstructing images...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Reconstructed images
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for (algo_name, data), pos in zip(algorithms.items(), positions):
        reconstructed = reconstruct_image(pixels, data['labels'], data['centroids'], original_shape)
        
        axes[pos].imshow(reconstructed)
        title = f"{algo_name}\n"
        title += f"Iterations: {data['iterations']}\n"
        title += f"Inertia ({data['space']}): {data['inertia']:.2e}"
        axes[pos].set_title(title, fontsize=11)
        axes[pos].axis('off')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.suptitle(f'K-Means Color Quantization Comparison (k={k})', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visual comparison saved to {save_path}")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for algo_name, data in algorithms.items():
        print(f"\n{algo_name.replace(chr(10), ' ')}:")
        print(f"  Iterations: {data['iterations']}")
        print(f"  Inertia ({data['space']}): {data['inertia']:.2f}")

if __name__ == "__main__":
    # Test on one image
    image_path = "../test_images/Test_image1.jpg"
    run_visual_comparison(image_path, k=K, save_path='../test_results/visual_comparison_k3_3.png')