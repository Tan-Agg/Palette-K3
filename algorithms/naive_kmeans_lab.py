import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    from algorithms.utils import load_image_as_array, rgb_to_lab, lab_to_rgb
except ImportError:
    from utils import load_image_as_array, rgb_to_lab, lab_to_rgb

K = 3    
class NaiveKMeansLAB:
    def __init__(self, k=K, max_iter=100, tol=0.01, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.iteration_data = []
        self.centroids_lab = None
        self.centroids_rgb = None
        self.labels = None
        self.inertia = None
        self.n_iterations = 0

    def random_initialization(self, pixels):
        """Pick k random pixels as initial centroids"""
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(pixels.shape[0], size=self.k, replace=False)
        return pixels[indices].astype(float)


    def fit(self, pixels_rgb):
        """
        Fit K-Means++ in LAB color space
        pixels_rgb: Nx3 array of RGB values (0-255)
        """
        # Convert RGB to LAB
        pixels_lab = rgb_to_lab(pixels_rgb)
        
        # Initialize centroids in LAB space
        self.centroids_lab = self.random_initialization(pixels_lab)
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids_lab.copy()
            
            # Assignment step (in LAB space)
            distances = np.linalg.norm(
                pixels_lab[:, np.newaxis, :] - self.centroids_lab[np.newaxis, :, :], 
                axis=-1
            )
            self.labels = np.argmin(distances, axis=-1)
            
            # Update step (in LAB space)
            self.inertia = 0
            for i in range(self.k):
                cluster_pixels = pixels_lab[self.labels == i]
                if len(cluster_pixels) > 0:
                    self.centroids_lab[i] = cluster_pixels.mean(axis=0)
                    self.inertia += ((cluster_pixels - self.centroids_lab[i])**2).sum()
            
            # Store iteration data
            self.iteration_data.append({
                'iteration': iteration + 1,
                'centroids': old_centroids.copy(),
                'inertia': self.inertia
            })
            
            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids_lab - old_centroids, ord='fro')
            
            if centroid_shift < self.tol:
                self.n_iterations = iteration + 1
                print(f"Converged after {self.n_iterations} iterations")
                break
        else:
            self.n_iterations = self.max_iter
            print(f"Reached max iterations ({self.max_iter})")
        
        # Convert final centroids back to RGB for display
        self.centroids_rgb = lab_to_rgb(self.centroids_lab)
        # store copy of original RGB pixels
        self.pixels_rgb_original = pixels_rgb.copy()
        # calculate inertia in RGB space for comparison
        self.inertia_rgb = self.calculate_rgb_inertia(pixels_rgb)
        
        return self
    
    def calculate_rgb_inertia(self, pixels_rgb):
        """
        Calculate inertia in RGB space for fair comparison with RGB algorithms
        pixels_rgb: original RGB pixels (Nx3)
        """
        # Convert LAB centroids back to RGB
        try:
            from algorithms.utils import lab_to_rgb
        except ImportError:
            from utils import lab_to_rgb
        centroids_rgb = lab_to_rgb(self.centroids_lab)
        
        # Calculate which centroid each pixel is closest to (in RGB space)
        distances = np.linalg.norm(
            pixels_rgb[:, np.newaxis, :] - centroids_rgb[np.newaxis, :, :],
            axis=-1
        )
        labels_rgb = np.argmin(distances, axis=-1)
        
        # Calculate inertia in RGB space
        inertia_rgb = 0
        for i in range(self.k):
            cluster_pixels = pixels_rgb[labels_rgb == i]
            if len(cluster_pixels) > 0:
                inertia_rgb += ((cluster_pixels - centroids_rgb[i])**2).sum()
        
        return inertia_rgb

    def get_palette(self):
        """Return final color palette as RGB integers"""
        return self.centroids_rgb.astype(int)
    
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
        plt.plot(iterations, inertias, marker='o', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (WCSS in LAB space)')
        plt.title('LAB K-Means++ Convergence')
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
        ax.set_title('Extracted Color Palette (LAB K-Means++)', fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load image
    image_path = "../test_images/Test_image1.jpg"

    pixels = load_image_as_array(image_path)[0]
    
    print(f"Image loaded: {pixels.shape[0]} pixels")
    
    # Run LAB K-Means++
    kmeans = NaiveKMeansLAB(k=K, max_iter=100, random_state=42)
    kmeans.fit(pixels)
    
    # Show results
    print(f"Final inertia (LAB space): {kmeans.inertia:.2f}")
    print(f"Final inertia (RGB space): {kmeans.inertia_rgb:.2f}")
    print(f"Color palette (RGB):")
    for i, color in enumerate(kmeans.get_palette()):
        print(f"  Color {i+1}: RGB{tuple(color)}")
    
    print(f"\nColor palette (HEX):")
    for i, hex_code in enumerate(kmeans.get_palette_hex()):
        print(f"  Color {i+1}: {hex_code}")
    
    # Visualize
    kmeans.visualize_convergence()
    kmeans.visualize_palette()