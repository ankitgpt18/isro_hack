import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def shape_from_shading(image, light_source_direction, n_iterations=100, learning_rate=1e-3):
    """
    Generates a depth map from a single image using Horn's iterative method (simplified).
    
    Args:
        image (np.array): The input grayscale image.
        light_source_direction (tuple): A (x, y, z) tuple for the light source vector.
        n_iterations (int): The number of iterations to run.
        learning_rate (float): The step size for each iteration.
        
    Returns:
        np.array: The estimated depth map.
    """
    # Normalize image to be between 0 and 1
    image = image.astype(np.float32) / 255.0
    
    # Initial depth map (Z) is a flat plane
    z = np.zeros(image.shape)
    
    # Get image gradients (p, q)
    p = np.zeros(image.shape)
    q = np.zeros(image.shape)

    # Light source vector (assuming it's coming from a direction)
    light_source = np.array(light_source_direction)
    light_source = light_source / np.linalg.norm(light_source)

    print("Starting Shape from Shading iterations...")
    for i in range(n_iterations):
        # Calculate gradients of the current depth map
        p[:, :-1] = z[:, 1:] - z[:, :-1]
        q[:-1, :] = z[1:, :] - z[:-1, :]
        
        # Reflectance model (simplified Lambertian)
        # R(p, q) = (1 + p*light_source[0] + q*light_source[1]) / sqrt(1 + p^2 + q^2)
        # We are trying to minimize (Image - R)^2
        # For simplicity, we'll use a simplified update rule.
        
        # This is a very simplified version of Horn's method.
        # A full implementation would be more complex.
        
        # Avoid division by zero
        denominator = np.sqrt(1 + p**2 + q**2)
        denominator[denominator == 0] = 1

        reflectance = (1 + p * light_source[0] + q * light_source[1]) / denominator
        error = image - reflectance
        
        # Update depth map based on the error
        # This is a gradient descent-like update
        update = np.zeros_like(z)
        update[:, :-1] += error[:, :-1] * light_source[0]
        update[:-1, :] += error[:-1, :] * light_source[1]
        
        z += learning_rate * update
        
        if (i+1) % 10 == 0:
            print(f"  Iteration {i+1}/{n_iterations}, Mean Error: {np.mean(np.abs(error)):.4f}")
            
    return z

def visualize_results(original_image, depth_map):
    """Displays the original image and the resulting depth map."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Lunar Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(depth_map, cmap='viridis')
    axes[1].set_title('Generated Depth Map (DEM)')
    axes[1].axis('off')
    
    fig.colorbar(im, ax=axes[1], label='Relative Height')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    INPUT_IMAGE = 'virtanen_crater.jpg'

    try:
        # Load the grayscale image
        image = Image.open(INPUT_IMAGE).convert('L')
        image_array = np.array(image)

        # --- Parameters for Shape from Shading ---
        # Since we don't have metadata, we'll guess the light direction.
        # Let's assume light is coming from the top-left.
        light_direction = (-1, -1, 0.5)
        
        # Generate the depth map
        depth_map = shape_from_shading(image_array, light_direction, n_iterations=200, learning_rate=1e-4)
        
        # Visualize
        visualize_results(image_array, depth_map)

    except FileNotFoundError:
        print(f"Error: Input image '{INPUT_IMAGE}' not found. Please run the download_image.py script first.")
    except Exception as e:
        print(f"An error occurred: {e}") 