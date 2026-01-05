import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from algorithms.utils import load_image_as_array, rgb_to_lab, lab_to_rgb

class KMeansPlusPlus:
    def __init__(self, k=5, max_iter=100, tol=0.01, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.iteration_data = []
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = 0
        self.inertia_lab = None  # Add this line

    def kmeans_plus_plus_initialization(self, pixels):
        """K-Means++ initialization using mask-based selection"""
        rng = np.random.RandomState(self.random_state)
        idx_mask = np.zeros(shape=pixels.shape[0]).astype('bool')
        data_idxs = np.arange(pixels.shape[0])
        
        # Choose first centroid randomly
        idx = rng.choice(data_idxs)
        idx_mask[idx] = True
        
        # Choose remaining k-1 centroids
        for _ in range(1, self.k):
            # Calculate min squared distance to nearest existing centroid
            min_centroid_dist = (np.linalg.norm(
                pixels[~idx_mask, np.newaxis, :] - pixels[np.newaxis, idx_mask, :],
                axis=-1
            )**2).min(axis=-1)
            
            # Probability proportional to D(x)^2
            centroid_prob = min_centroid_dist / min_centroid_dist.sum()
            idx = rng.choice(data_idxs[~idx_mask], p=centroid_prob)
            idx_mask[idx] = True
        
        return pixels[idx_mask].astype(float)

    def fit(self, pixels):
        """Fit K-Means++ to pixel data"""
        self.centroids = self.kmeans_plus_plus_initialization(pixels)
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Assignment step
            distances = np.linalg.norm(
                pixels[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], 
                axis=-1
            )
            self.labels = np.argmin(distances, axis=-1)
            
            # Update step
            self.inertia = 0
            for i in range(self.k):
                cluster_pixels = pixels[self.labels == i]
                if len(cluster_pixels) > 0:
                    self.centroids[i] = cluster_pixels.mean(axis=0)
                    self.inertia += ((cluster_pixels - self.centroids[i])**2).sum()
            
            # Store iteration data
            self.iteration_data.append({
                'iteration': iteration + 1,
                'centroids': old_centroids.copy(),
                'inertia': self.inertia
            })
            
            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids, ord='fro')
            
            if centroid_shift < self.tol:
                self.n_iterations = iteration + 1
                print(f"Converged after {self.n_iterations} iterations")
                break
        else:
            self.n_iterations = self.max_iter
            print(f"Reached max iterations ({self.max_iter})")
        
        # Store original RGB pixels for LAB inertia calculation
        self.pixels_rgb_original = pixels.copy()
        # Calculate inertia in LAB space for comparison
        self.inertia_lab = self.calculate_lab_inertia(pixels)
        
        
        return self
    


    def calculate_lab_inertia(self, pixels_rgb):
        """
        Calculate inertia in LAB space for fair comparison with LAB algorithms
        pixels_rgb: original RGB pixels (Nx3)
        """
        from algorithms.utils import rgb_to_lab
        
        # Convert RGB centroids and pixels to LAB
        centroids_lab = rgb_to_lab(self.centroids)
        pixels_lab = rgb_to_lab(pixels_rgb)
        
        # Calculate which centroid each pixel is closest to (in LAB space)
        distances = np.linalg.norm(
            pixels_lab[:, np.newaxis, :] - centroids_lab[np.newaxis, :, :],
            axis=-1
        )
        labels_lab = np.argmin(distances, axis=-1)
        
        # Calculate inertia in LAB space
        inertia_lab = 0
        for i in range(self.k):
            cluster_pixels = pixels_lab[labels_lab == i]
            if len(cluster_pixels) > 0:
                inertia_lab += ((cluster_pixels - centroids_lab[i])**2).sum()
        
        return inertia_lab


    def get_palette(self):
        """Return final color palette as RGB integers"""
        return np.round(self.centroids).astype(int)
    
    def get_palette_hex(self):
        """Return color palette as hex codes"""
        palette = self.get_palette()
        hex_codes = []
        for color in palette:
            hex_code = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            hex_codes.append(hex_code)
        return hex_codes

    def visualize_convergence(self):
        """Plot inertia over iterations"""
        iterations = [d['iteration'] for d in self.iteration_data]
        inertias = [d['inertia'] for d in self.iteration_data]
        
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, inertias, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (WCSS)')
        plt.title('K-Means++ Convergence')
        plt.grid(True)
        plt.show()

    def visualize_palette(self):
        """Display the extracted color palette with hex codes"""
        palette = self.get_palette()
        hex_codes = self.get_palette_hex()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        
        for i, (color, hex_code) in enumerate(zip(palette, hex_codes)):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color/255))
            ax.text(i + 0.5, 0.5, hex_code, 
                    ha='center', va='center', 
                    fontsize=10, fontweight='bold',
                    color='white' if np.mean(color) < 128 else 'black')
        
        ax.set_xlim(0, self.k)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Extracted Color Palette (K-Means++)', fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load image
    image_path = "../test_images/Test_image1.jpg"
    pixels = load_image_as_array(image_path)
    
    print(f"Image loaded: {pixels.shape[0]} pixels")
    
    # Run K-Means++
    kmeans = KMeansPlusPlus(k=5, max_iter=100, random_state=42)
    kmeans.fit(pixels)
    
    # Show results
    print(f"Final inertia (RGB space): {kmeans.inertia:.2f}")
    print(f"Final inertia (LAB space): {kmeans.inertia_lab:.2f}")
    print(f"Color palette (RGB):")
    for i, color in enumerate(kmeans.get_palette()):
        print(f"  Color {i+1}: RGB{tuple(color)}")
    
    print(f"\nColor palette (HEX):")
    for i, hex_code in enumerate(kmeans.get_palette_hex()):
        print(f"  Color {i+1}: {hex_code}")
    
    # Visualize
    kmeans.visualize_convergence()
    kmeans.visualize_palette()