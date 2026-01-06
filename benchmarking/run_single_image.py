import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


# Algorithms
from algorithms.naive_kmeans import NaiveKMeans
from algorithms.naive_kmeans_lab import NaiveKMeansLAB
from algorithms.kmeans_plusplus import KMeansPlusPlus
from algorithms.lab_kmeans import LABKMeansPlusPlus

from algorithms.utils import load_image_as_array


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def reconstruct_image(pixels, labels, centroids, image_shape):
    """Reconstruct quantized image from labels and centroids"""
    quantized = centroids[labels].astype(np.uint8)
    return quantized.reshape(image_shape)


def run_algorithm(name, model, pixels):
    """Run algorithm, measure runtime, collect metrics"""
    start = time.time()
    model.fit(pixels)
    runtime = time.time() - start

    result = {
        "name": name,
        "model": model,
        "runtime": runtime,
        "iterations": model.n_iterations,
        "palette": model.get_palette(),
        "labels": model.labels
    }

    # Unified inertia reporting
    result["inertia_rgb"] = getattr(model, "inertia_rgb", model.inertia)
    result["inertia_lab"] = getattr(model, "inertia_lab", model.inertia)

    return result


# -------------------------------------------------------
# Main benchmark
# -------------------------------------------------------

def main():
    #ATH = "../test_images/Test_image1.jpg"
    IMAGE_PATH = os.path.join(PROJECT_ROOT, "test_images", "Test_image1.jpg")
    K = 10
    MAX_ITER = 100
    RANDOM_STATE = 42

    # Load image
    pixels, img, image_shape = load_image_as_array(IMAGE_PATH)
    # Infer image shape from flattened pixels
    # num_pixels = pixels.shape[0]
    # image_shape = (int(np.sqrt(num_pixels)), int(num_pixels / int(np.sqrt(num_pixels))), 3)


    print(f"Loaded image with {pixels.shape[0]} pixels")

    # Define algorithms
    algorithms = [
        ("Naive K-Means (RGB)", NaiveKMeans(k=K, max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ("K-Means++ (RGB)", KMeansPlusPlus(k=K, max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ("Naive K-Means (LAB)", NaiveKMeansLAB(k=K, max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ("K-Means++ (LAB)", LABKMeansPlusPlus(k=K, max_iter=MAX_ITER, random_state=RANDOM_STATE)),
    ]

    results = []

    # Run all algorithms
    for name, model in algorithms:
        print(f"\nRunning {name}...")
        results.append(run_algorithm(name, model, pixels))

    # ---------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------

    print("\n================ Benchmark Results ================\n")
    header = f"{'Method':30s} {'Iter':>6s} {'RGB Inertia':>15s} {'LAB Inertia':>15s} {'Time (s)':>10s}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['name']:30s} "
            f"{r['iterations']:6d} "
            f"{r['inertia_rgb']:15.2f} "
            f"{r['inertia_lab']:15.2f} "
            f"{r['runtime']:10.3f}"
        )

    # ---------------------------------------------------
    # Visualization grid
    # ---------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Input Image (Resized)")
    axes[0].axis("off")

    # Quantized images
    for i, r in enumerate(results, start=1):
        quantized_img = reconstruct_image(
            pixels,
            r["labels"],
            r["palette"],
            image_shape
        )

        axes[i].imshow(quantized_img)
        axes[i].set_title(r["name"])
        axes[i].axis("off")

    # Hide unused subplot
    axes[-1].axis("off")

    plt.suptitle("Color Quantization Comparison (k=10)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # Palette comparison
    # ---------------------------------------------------

    fig, axes = plt.subplots(len(results), 1, figsize=(12, 8))

    for ax, r in zip(axes, results):
        palette = r["palette"]
        for i, color in enumerate(palette):
            ax.add_patch(
                plt.Rectangle((i, 0), 1, 1, color=color / 255)
            )
        ax.set_xlim(0, K)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r["name"])

    plt.suptitle("Extracted Color Palettes", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
