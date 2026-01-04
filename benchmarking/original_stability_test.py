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
        'Naive K-Means (RGB)': {'iterations': [], 'inertia': []},  # RENAME
        'K-Means++ (RGB)': {'iterations': [], 'inertia': []},      # RENAME
        'Naive K-Means (LAB)': {'iterations': [], 'inertia': []},  # ADD THIS
        'K-Means++ (LAB)': {'iterations': [], 'inertia': []}       # RENAME
    }
    
    # Run tests
    for seed in range(n_runs):
        print(f"Run {seed + 1}/{n_runs}...", end=' ')
        
        # Naive K-Means (RGB)
        naive = NaiveKMeans(k=k, random_state=seed, max_iter=100)
        naive.fit(pixels)
        results['Naive K-Means (RGB)']['iterations'].append(naive.n_iterations)
        results['Naive K-Means (RGB)']['inertia'].append(naive.inertia)
        
        # K-Means++ (RGB)
        kpp = KMeansPlusPlus(k=k, random_state=seed, max_iter=100)
        kpp.fit(pixels)
        results['K-Means++ (RGB)']['iterations'].append(kpp.n_iterations)
        results['K-Means++ (RGB)']['inertia'].append(kpp.inertia)
        
        # ADD THIS BLOCK:
        # Naive K-Means (LAB)
        naive_lab = NaiveKMeansLAB(k=k, random_state=seed, max_iter=100)
        naive_lab.fit(pixels)
        results['Naive K-Means (LAB)']['iterations'].append(naive_lab.n_iterations)
        results['Naive K-Means (LAB)']['inertia'].append(naive_lab.inertia)
        
        # K-Means++ (LAB)
        lab = LABKMeansPlusPlus(k=k, random_state=seed, max_iter=100)
        lab.fit(pixels)
        results['K-Means++ (LAB)']['iterations'].append(lab.n_iterations)
        results['K-Means++ (LAB)']['inertia'].append(lab.inertia)
        
        print("Done")
    
    return results

def print_statistics(results):
    """Print mean and std deviation for each algorithm"""
    print("\n" + "="*60)
    print("STABILITY TEST RESULTS")
    print("="*60)
    
    for algo_name, data in results.items():
        iterations = np.array(data['iterations'])
        inertias = np.array(data['inertia'])
        
        print(f"\n{algo_name}:")
        print(f"  Iterations: {list(iterations)}")
        print(f"  Mean: {iterations.mean():.2f} ± {iterations.std():.2f}")
        print(f"  Range: [{iterations.min()}, {iterations.max()}]")
        print(f"  Inertia mean: {inertias.mean():.2f} ± {inertias.std():.2f}")

def visualize_results(results):
    """Create box plots comparing algorithms"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Make wider
    
    # Iterations comparison - ALL 4 NOW
    iterations_data = [
        results['Naive K-Means (RGB)']['iterations'],
        results['K-Means++ (RGB)']['iterations'],
        results['Naive K-Means (LAB)']['iterations'],  # ADD THIS
        results['K-Means++ (LAB)']['iterations']
    ]
    
    ax1.boxplot(iterations_data, tick_labels=['Naive\n(RGB)', 'K-Means++\n(RGB)', 
                                               'Naive\n(LAB)', 'K-Means++\n(LAB)'])
    ax1.set_ylabel('Iterations to Convergence', fontsize=12)
    ax1.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add mean values
    for i, data in enumerate(iterations_data, 1):
        mean_val = np.mean(data)
        ax1.text(i, mean_val, f'{mean_val:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Inertia comparison (RGB only)
    inertia_data = [
        results['Naive K-Means (RGB)']['inertia'],
        results['K-Means++ (RGB)']['inertia']
    ]
    
    ax2.boxplot(inertia_data, tick_labels=['Naive\n(RGB)', 'K-Means++\n(RGB)'])
    ax2.set_ylabel('Final Inertia (RGB space)', fontsize=12)
    ax2.set_title('Clustering Quality Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('stability_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved as 'stability_comparison.png'")
    plt.show()

# Rest stays the same

if __name__ == "__main__":
    # Run stability test
    image_path = "../test_images/Test_image1.jpg"

    results = run_stability_test(image_path, k=5, n_runs=20)
    
    # Print statistics
    print_statistics(results)
    
    # Visualize
    visualize_results(results)