import numpy as np

def horn_sfs(image, light_source, n_iterations=500, lambd=0.2, learning_rate=1e-3):
    """
    Advanced Horn's Shape-from-Shading with regularization.
    Args:
        image (np.array): Grayscale image, normalized [0,1].
        light_source (tuple): (lx, ly, lz) direction, normalized.
        n_iterations (int): Number of iterations.
        lambd (float): Regularization weight.
        learning_rate (float): Step size.
    Returns:
        np.array: Estimated depth map.
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    z = np.zeros_like(image)
    p = np.zeros_like(image)
    q = np.zeros_like(image)
    lx, ly, lz = light_source / np.linalg.norm(light_source)
    for it in range(n_iterations):
        # Compute gradients
        p[:, :-1] = z[:, 1:] - z[:, :-1]
        q[:-1, :] = z[1:, :] - z[:-1, :]
        # Lambertian reflectance
        denom = np.sqrt(1 + p**2 + q**2)
        denom[denom == 0] = 1
        R = (lx * p + ly * q + lz) / denom
        # Data term
        error = image - R
        # Regularization (smoothness)
        z_avg = (np.roll(z, 1, axis=0) + np.roll(z, -1, axis=0) + np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1)) / 4
        reg = lambd * (z_avg - z)
        # Update
        update = np.zeros_like(z)
        update[:, :-1] += error[:, :-1] * lx
        update[:-1, :] += error[:-1, :] * ly
        update += reg
        z += learning_rate * update
        if (it+1) % 50 == 0:
            print(f"Iteration {it+1}/{n_iterations}, Mean Error: {np.mean(np.abs(error)):.4f}")
    return z 