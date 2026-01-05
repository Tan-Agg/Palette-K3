import matplotlib.pyplot as plt
import numpy as np
import os
import re

def parse_metrics_file(filepath):
    """Parse a single metrics file and extract mean and std values"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Extract data for each algorithm
    algorithms = [
        'Naive K-Means (RGB)',
        'K-Means++ (RGB)',
        'Naive K-Means (LAB)',
        'K-Means++ (LAB)'
    ]
    
    for algo in algorithms:
        # Find the section for this algorithm
        pattern = f"{re.escape(algo)}:.*?Mean Iterations: ([\d.]+) ± ([\d.]+).*?Mean Inertia \(RGB\): ([\d.]+) ± ([\d.]+).*?Mean Inertia \(LAB\): ([\d.]+) ± ([\d.]+)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            results[algo] = {
                'iter_mean': float(match.group(1)),
                'iter_std': float(match.group(2)),
                'inertia_rgb_mean': float(match.group(3)),
                'inertia_rgb_std': float(match.group(4)),
                'inertia_lab_mean': float(match.group(5)),
                'inertia_lab_std': float(match.group(6))
            }
    
    return results

def collect_all_results(results_folder='test_results'):
    """Collect results from all metrics files"""
    all_results = {}
    
    # Get all metrics files
    files = sorted([f for f in os.listdir(results_folder) if f.endswith('_metrics.txt')])
    
    for filename in files:
        filepath = os.path.join(results_folder, filename)
        image_name = filename.replace('_metrics.txt', '')
        all_results[image_name] = parse_metrics_file(filepath)
    
    return all_results

def prepare_plot_data(all_results):
    """Convert results dict to arrays for plotting"""
    images = sorted(all_results.keys())
    n_images = len(images)
    
    algorithms = [
        'Naive K-Means (RGB)',
        'K-Means++ (RGB)',
        'Naive K-Means (LAB)',
        'K-Means++ (LAB)'
    ]
    
    # Initialize arrays
    iterations_mean = np.zeros((n_images, 4))
    iterations_std = np.zeros((n_images, 4))
    inertia_rgb_mean = np.zeros((n_images, 4))
    inertia_rgb_std = np.zeros((n_images, 4))
    inertia_lab_mean = np.zeros((n_images, 4))
    inertia_lab_std = np.zeros((n_images, 4))
    
    for i, img in enumerate(images):
        for j, algo in enumerate(algorithms):
            data = all_results[img][algo]
            iterations_mean[i, j] = data['iter_mean']
            iterations_std[i, j] = data['iter_std']
            inertia_rgb_mean[i, j] = data['inertia_rgb_mean']
            inertia_rgb_std[i, j] = data['inertia_rgb_std']
            inertia_lab_mean[i, j] = data['inertia_lab_mean']
            inertia_lab_std[i, j] = data['inertia_lab_std']
    
    return images, iterations_mean, iterations_std, inertia_rgb_mean, inertia_rgb_std, inertia_lab_mean, inertia_lab_std

def plot_results(images, iterations_mean, iterations_std, 
                 inertia_rgb_mean, inertia_rgb_std, 
                 inertia_lab_mean, inertia_lab_std,
                 save_path='combined_results.png'):
    """Create three subplots: iterations, RGB inertia, LAB inertia"""
    
    methods = ['Naive', 'K-Means++']
    colors = ['skyblue', 'orange', 'lightgreen', 'red']
    labels = ['Naive (RGB)', 'K++ (RGB)', 'Naive (LAB)', 'K++ (LAB)']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Plot 1: Iterations ---
    for i in range(4):
        ax1.errorbar(images, iterations_mean[:, i], yerr=iterations_std[:, i], 
                     fmt='-o', color=colors[i], label=labels[i], linewidth=2, markersize=6)
    
    ax1.set_title('Convergence Speed (Iterations)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Iterations ± Std', fontsize=12)
    ax1.set_xlabel('Image', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # --- Plot 2: RGB Inertia ---
    for i in range(4):
        ax2.errorbar(images, inertia_rgb_mean[:, i], yerr=inertia_rgb_std[:, i], 
                     fmt='-o', color=colors[i], label=labels[i], linewidth=2, markersize=6)
    
    ax2.set_title('Clustering Quality - RGB Space', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Inertia (RGB) ± Std', fontsize=12)
    ax2.set_xlabel('Image', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # --- Plot 3: LAB Inertia ---
    for i in range(4):
        ax3.errorbar(images, inertia_lab_mean[:, i], yerr=inertia_lab_std[:, i], 
                     fmt='-o', color=colors[i], label=labels[i], linewidth=2, markersize=6)
    
    ax3.set_title('Clustering Quality - LAB Space (Perceptual)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Inertia (LAB) ± Std', fontsize=12)
    ax3.set_xlabel('Image', fontsize=12)
    ax3.set_yscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, which="both", ls="--", alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Combined plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Collect all results
    print("Parsing metrics files...")
    all_results = collect_all_results('test_results')
    print(f"Found {len(all_results)} images")
    
    # Prepare data for plotting
    images, iterations_mean, iterations_std, \
    inertia_rgb_mean, inertia_rgb_std, \
    inertia_lab_mean, inertia_lab_std = prepare_plot_data(all_results)
    
    # Create plots
    plot_results(images, iterations_mean, iterations_std,
                 inertia_rgb_mean, inertia_rgb_std,
                 inertia_lab_mean, inertia_lab_std)