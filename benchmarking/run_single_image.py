import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path to import from algorithms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.naive_kmeans import NaiveKMeans
from algorithms.kmeans_plusplus import KMeansPlusPlus
from algorithms.naive_kmeans_lab import NaiveKMeansLAB
from algorithms.lab_kmeans import LABKMeansPlusPlus
from algorithms.utils import load_image_as_array

def run_single_image_comparison(image_path, k=8, max_iterations=100):
    """
    Run all 4 K-Means variants on a single image and compare results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"K={k}, Max Iterations={max_iterations}")
    print(f"{'='*60}\n")
    
    # Load image
    pixels_rgb = load_image_as_array(image_path)
    print(f"Image loaded: {pixels_rgb.shape[0]} pixels")
    
    results = {}
    
    # 1. Naive K-Means (RGB)
    print("\n[1/4] Running Naive K-Means (RGB)...")
    kmeans_naive_rgb = NaiveKMeans(k=k, max_iterations=max_iterations)
    kmeans_naive_rgb.fit(pixels_rgb)
    results['Naive (RGB)'] = {
        'iterations': kmeans_naive_rgb.iterations,
        'inertia_rgb': kmeans_naive_rgb.inertia,
        'inertia_lab': kmeans_naive_rgb.calculate_lab_inertia(pixels_rgb),
        'centroids': kmeans_naive_rgb.centroids
    }
    print(f"   Converged in {kmeans_naive_rgb.iterations} iterations")
    
    # 2. K-Means++ (RGB)
    print("\n[2/4] Running K-Means++ (RGB)...")
    kmeans_pp_rgb = KMeansPlusPlus(k=k, max_iterations=max_iterations)
    kmeans_pp_rgb.fit(pixels_rgb)
    results['K++ (RGB)'] = {
        'iterations': kmeans_pp_rgb.iterations,
        'inertia_rgb': kmeans_pp_rgb.inertia,
        'inertia_lab': kmeans_pp_rgb.calculate_lab_inertia(pixels_rgb),
        'centroids': kmeans_pp_rgb.centroids
    }
    print(f"   Converged in {kmeans_pp_rgb.iterations} iterations")
    
    # 3. Naive K-Means (LAB)
    print("\n[3/4] Running Naive K-Means (LAB)...")
    kmeans_naive_lab = NaiveKMeansLAB(k=k, max_iterations=max_iterations)
    kmeans_naive_lab.fit(pixels_rgb)
    results['Naive (LAB)'] = {
        'iterations': kmeans_naive_lab.iterations,
        'inertia_rgb': kmeans_naive_lab.calculate_rgb_inertia(pixels_rgb),
        'inertia_lab': kmeans_naive_lab.inertia_lab,
        'centroids': kmeans_naive_lab.centroids_lab
    }
    print(f"   Converged in {kmeans_naive_lab.iterations} iterations")
    
    # 4. K-Means++ (LAB)
    print("\n[4/4] Running K-Means++ (LAB)...")
    kmeans_pp_lab = LABKMeansPlusPlus(k=k, max_iterations=max_iterations)
    kmeans_pp_lab.fit(pixels_rgb)
    results['K++ (LAB)'] = {
        'iterations': kmeans_pp_lab.iterations,
        'inertia_rgb': kmeans_pp_lab.calculate_rgb_inertia(pixels_rgb),
        'inertia_lab': kmeans_pp_lab.inertia_lab,
        'centroids': kmeans_pp_lab.centroids_lab
    }
    print(f"   Converged in {kmeans_pp_lab.iterations} iterations")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Iterations':<12} {'Inertia (RGB)':<18} {'Inertia (LAB)':<18}")
    print(f"{'-'*60}")
    
    for method, data in results.items():
        print(f"{method:<15} {data['iterations']:<12} {data['inertia_rgb']:<18.2e} {data['inertia_lab']:<18.2e}")
    
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    image_path = "../test_images/Test_image1.jpg"  # Adjust path as needed
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Usage: python run_single_image.py <path_to_image>")
        sys.exit(1)
    
    results = run_single_image_comparison(
        image_path=image_path,
        k=8,
        max_iter=100
    )