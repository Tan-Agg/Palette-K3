from PIL import Image
import numpy as np

def load_image_as_array(image_path):
    """
    Load an image from the specified path and convert it to a numpy array of shape (N, 3),
    where N is the number of pixels and each pixel is represented by its RGB values.
    
    Parameters:
    - image_path: str, path to the image file.
    
    Returns:
    - pixels: numpy array of shape (N, 3)
    """
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Reshape the array to (N, 3)
    pixels = image_array.reshape(-1, 3)
    
    return pixels
# load_image_as_array(image_path)
# Should use PIL to open image
# Convert to RGB (in case it's RGBA or grayscale)
# Convert to numpy array
# Reshape from (height, width, 3) to (height*width, 3) - flatten to list of pixels
# Return the Nx3 array


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two RGB points.
    
    Parameters:
    - point1: numpy array of shape (3,), representing the first RGB point.
    - point2: numpy array of shape (3,), representing the second RGB point.
    
    Returns:
    - distance: float, the Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))
# euclidean_distance(point1, point2)
# euclidean_distance(point1, point2)

# Take two RGB points (each is array of 3 values)
# Calculate sqrt(sum of squared differences)
# Return the distance

def calculate_inertia(pixels, centroids, assignments):
    """
    Calculate the inertia (sum of squared distances to nearest centroid) for the given pixel assignments.
    
    Parameters:
    - pixels: numpy array of shape (N, 3), the pixel data.
    - centroids: numpy array of shape (k, 3), the centroid colors.
    - assignments: numpy array of shape (N,), the index of the assigned centroid for each pixel.
    
    Returns:
    - inertia: float, the total inertia value.
    """
    inertia = 0.0
    for i in range(pixels.shape[0]):
        centroid = centroids[assignments[i]]
        inertia += np.sum((pixels[i] - centroid) ** 2)
    return inertia
# calculate_inertia(pixels, centroids, assignments)
# For each pixel, find its assigned centroid using assignments array
# Calculate squared distance to that centroid
# Sum all squared distances
# Return total inertia