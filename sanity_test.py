# run_all_stability_tests.py
import os
import pickle
from benchmarking.stability_test import run_stability_test, print_statistics, visualize_results

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_FOLDER = "test_images"
RESULTS_FOLDER = "test_results"
K = 5
N_RUNS = 20
# ----------------------------

# Create results folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Get all image files in test_images folder
image_files = [f for f in os.listdir(IMAGE_FOLDER) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

for img_file in image_files:
    image_path = os.path.join(IMAGE_FOLDER, img_file)
    print(f"\n=== Processing {img_file} ===\n")
    
    # Run stability test
    results = run_stability_test(image_path, k=K, n_runs=N_RUNS)
    
    # ----------------------------
    # Save metrics to txt file
    # ----------------------------
    txt_path = os.path.join(RESULTS_FOLDER, img_file + "_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Stability Test Metrics for {img_file}\n")
        f.write("="*60 + "\n")
        for algo_name, data in results.items():
            iterations = data['iterations']
            inertias_rgb = data['inertia_rgb']
            inertias_lab = data['inertia_lab']
            
            f.write(f"\n{algo_name}:\n")
            f.write(f"Iterations: {iterations}\n")
            f.write(f"Mean Iterations: {sum(iterations)/len(iterations):.2f} ± {((sum((x - sum(iterations)/len(iterations))**2 for x in iterations)/len(iterations))**0.5):.2f}\n")
            
            f.write(f"Inertia (RGB): {inertias_rgb}\n")
            f.write(f"Mean Inertia (RGB): {sum(inertias_rgb)/len(inertias_rgb):.2f} ± {((sum((x - sum(inertias_rgb)/len(inertias_rgb))**2 for x in inertias_rgb)/len(inertias_rgb))**0.5):.2f}\n")
            
            f.write(f"Inertia (LAB): {inertias_lab}\n")
            f.write(f"Mean Inertia (LAB): {sum(inertias_lab)/len(inertias_lab):.2f} ± {((sum((x - sum(inertias_lab)/len(inertias_lab))**2 for x in inertias_lab)/len(inertias_lab))**0.5):.2f}\n")
        f.write("\n")
    print(f"✅ Metrics saved to {txt_path}")
    
    # ----------------------------
    # Save plot
    # ----------------------------
    visualize_results(results, save_path=os.path.join(RESULTS_FOLDER, img_file + "_plot.png"))
    print(f"✅ Plot saved to {os.path.join(RESULTS_FOLDER, img_file + '_plot.png')}")
    
print("\nAll images processed successfully!")