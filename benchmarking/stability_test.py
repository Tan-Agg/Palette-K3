import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from algorithms.naive_kmeans import NaiveKMeans
from algorithms.kmeans_plusplus import KMeansPlusPlus
from algorithms.lab_kmeans import LABKMeansPlusPlus
from algorithms.naive_kmeans_lab import NaiveKMeansLAB
from algorithms.utils import load_image_as_array


def run_stability_test(image_path, k=5, n_runs=20):
    """
    Run each algorithm multiple times with different seeds
    Track iterations and inertia for each run
    """
    print(f"Running stability test on {image_path}")
    print(f"k={k}, runs={n_runs}\n")
    
    # Load image once
    pixels = load_image_as_array(image_path)
    print(f"Image loaded: {pixels.shape[0]} pixels\n")
    
    results = {
    'Naive K-Means (RGB)': {'iterations': [], 'inertia_rgb': [], 'inertia_lab': []},
    'K-Means++ (RGB)': {'iterations': [], 'inertia_rgb': [], 'inertia_lab': []},
    'Naive K-Means (LAB)': {'iterations': [], 'inertia_rgb': [], 'inertia_lab': []},
    'K-Means++ (LAB)': {'iterations': [], 'inertia_rgb': [], 'inertia_lab': []}
}
    
    # Run tests
    for seed in range(n_runs):
        print(f"Run {seed + 1}/{n_runs}...", end=' ')
        
        # Naive K-Means (RGB)
        naive = NaiveKMeans(k=k, random_state=seed, max_iter=100)
        naive.fit(pixels)
        results['Naive K-Means (RGB)']['iterations'].append(naive.n_iterations)
        results['Naive K-Means (RGB)']['inertia_rgb'].append(naive.inertia)
        results['Naive K-Means (RGB)']['inertia_lab'].append(naive.inertia_lab)

        # K-Means++ (RGB)
        kpp = KMeansPlusPlus(k=k, random_state=seed, max_iter=100)
        kpp.fit(pixels)
        results['K-Means++ (RGB)']['iterations'].append(kpp.n_iterations)
        results['K-Means++ (RGB)']['inertia_rgb'].append(kpp.inertia)
        results['K-Means++ (RGB)']['inertia_lab'].append(kpp.inertia_lab)

        # Naive K-Means (LAB)
        naive_lab = NaiveKMeansLAB(k=k, random_state=seed, max_iter=100)
        naive_lab.fit(pixels)
        results['Naive K-Means (LAB)']['iterations'].append(naive_lab.n_iterations)
        results['Naive K-Means (LAB)']['inertia_rgb'].append(naive_lab.inertia_rgb)
        results['Naive K-Means (LAB)']['inertia_lab'].append(naive_lab.inertia)

        # K-Means++ (LAB)
        lab = LABKMeansPlusPlus(k=k, random_state=seed, max_iter=100)
        lab.fit(pixels)
        results['K-Means++ (LAB)']['iterations'].append(lab.n_iterations)
        results['K-Means++ (LAB)']['inertia_rgb'].append(lab.inertia_rgb)
        results['K-Means++ (LAB)']['inertia_lab'].append(lab.inertia)
        
        print("Done")
    
    return results


def print_statistics(results):
    """Print mean and std deviation for each algorithm"""
    print("\n" + "="*60)
    print("STABILITY TEST RESULTS")
    print("="*60)
    
    for algo_name, data in results.items():
        iterations = np.array(data['iterations'])
        inertias_rgb = np.array(data['inertia_rgb'])
        inertias_lab = np.array(data['inertia_lab'])
        
        print(f"\n{algo_name}:")
        print(f"  Iterations: {iterations.mean():.2f} ± {iterations.std():.2f}")
        print(f"  Range: [{iterations.min()}, {iterations.max()}]")
        print(f"  Inertia (RGB): {inertias_rgb.mean():.2f} ± {inertias_rgb.std():.2f}")
        print(f"  Inertia (LAB): {inertias_lab.mean():.2f} ± {inertias_lab.std():.2f}")


def visualize_results(results, save_path="stability_comparison.png"):
    """Create box plots comparing algorithms"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Iterations comparison
    iterations_data = [
        results['Naive K-Means (RGB)']['iterations'],
        results['K-Means++ (RGB)']['iterations'],
        results['Naive K-Means (LAB)']['iterations'],
        results['K-Means++ (LAB)']['iterations']
    ]
    
    ax1.boxplot(iterations_data, tick_labels=['Naive\n(RGB)', 'K-Means++\n(RGB)', 
                                               'Naive\n(LAB)', 'K-Means++\n(LAB)'])
    ax1.set_ylabel('Iterations to Convergence', fontsize=12)
    ax1.set_title('Convergence Speed', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for i, data in enumerate(iterations_data, 1):
        mean_val = np.mean(data)
        ax1.text(i, mean_val, f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Inertia in RGB space
    inertia_rgb_data = [
        results['Naive K-Means (RGB)']['inertia_rgb'],
        results['K-Means++ (RGB)']['inertia_rgb'],
        results['Naive K-Means (LAB)']['inertia_rgb'],
        results['K-Means++ (LAB)']['inertia_rgb']
    ]
    
    ax2.boxplot(inertia_rgb_data, tick_labels=['Naive\n(RGB)', 'K-Means++\n(RGB)', 
                                                'Naive\n(LAB)', 'K-Means++\n(LAB)'])
    ax2.set_ylabel('Final Inertia (RGB space)', fontsize=12)
    ax2.set_title('Clustering Quality - RGB Space', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. Inertia in LAB space
    inertia_lab_data = [
        results['Naive K-Means (RGB)']['inertia_lab'],
        results['K-Means++ (RGB)']['inertia_lab'],
        results['Naive K-Means (LAB)']['inertia_lab'],
        results['K-Means++ (LAB)']['inertia_lab']
    ]
    
    ax3.boxplot(inertia_lab_data, tick_labels=['Naive\n(RGB)', 'K-Means++\n(RGB)', 
                                                'Naive\n(LAB)', 'K-Means++\n(LAB)'])
    ax3.set_ylabel('Final Inertia (LAB space)', fontsize=12)
    ax3.set_title('Clustering Quality - LAB Space (Perceptual)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Visualization saved as '{save_path}'")
    return save_path


if __name__ == "__main__":
    # Example run for a single image
    image_path = "../test_images/Test_image1.jpg"
    results = run_stability_test(image_path, k=5, n_runs=20)
    print_statistics(results)
    visualize_results(results)
