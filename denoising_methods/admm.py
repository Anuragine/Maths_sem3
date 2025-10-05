import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# === PARAMETERS ===
lambda_tv = 0.05      # regularization strength (increased for better edge preservation)
rho = 1.0            # ADMM penalty parameter (increased)
iterations = 200       # number of ADMM iterations
input_folder = 'input_imgs'
output_folder = 'denoised_admm'
os.makedirs(output_folder, exist_ok=True)

# === HELPER FUNCTIONS ===
def gradient(img):
    """Compute forward differences for gradient with proper boundary handling"""
    grad_x = np.zeros_like(img, dtype=np.float64)
    grad_y = np.zeros_like(img, dtype=np.float64)
    
    # Forward differences with zero boundary conditions
    grad_x[:, :-1] = img[:, 1:] - img[:, :-1]
    grad_y[:-1, :] = img[1:, :] - img[:-1, :]
    
    return grad_x, grad_y

def divergence(grad_x, grad_y):
    """Compute divergence as adjoint of gradient operator"""
    div = np.zeros_like(grad_x, dtype=np.float64)
    
    # Adjoint of forward differences (backward differences)
    div[:, 0] = grad_x[:, 0]                    # First column
    div[:, 1:-1] = grad_x[:, 1:-1] - grad_x[:, :-2]  # Middle columns
    div[:, -1] = -grad_x[:, -2]                 # Last column
    
    temp = np.zeros_like(grad_y, dtype=np.float64)
    temp[0, :] = grad_y[0, :]                   # First row
    temp[1:-1, :] = grad_y[1:-1, :] - grad_y[:-2, :]  # Middle rows
    temp[-1, :] = -grad_y[-2, :]                # Last row
    
    div += temp
    return div

def shrink(x, thresh):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

def solve_x_direct(b, rho, shape):
    """Solve (I + rho*div*grad)x = b using direct method"""
    rows, cols = shape
    
    # Create the system matrix eigenvalues for discrete Laplacian
    # Using proper eigenvalues for forward-backward difference operator
    i_vals = np.arange(rows).reshape(-1, 1)
    j_vals = np.arange(cols).reshape(1, -1)
    
    # Eigenvalues for the discrete Laplacian with zero boundary conditions
    lambda_i = 2 * (1 - np.cos(np.pi * i_vals / rows))
    lambda_j = 2 * (1 - np.cos(np.pi * j_vals / cols))
    eigenvals = lambda_i + lambda_j
    
    # System matrix: I + rho * L
    system_eigenvals = 1 + rho * eigenvals
    system_eigenvals[0, 0] = 1  # Handle DC component
    
    # Solve using DCT (more appropriate for zero boundary conditions)
    from scipy.fftpack import dct, idct
    
    # Apply 2D DCT
    b_dct = dct(dct(b, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Solve in frequency domain
    x_dct = b_dct / system_eigenvals
    
    # Apply inverse 2D DCT
    x = idct(idct(x_dct, axis=1, norm='ortho'), axis=0, norm='ortho')
    
    return x

# === ADMM DENOISING FUNCTION ===
def admm_denoise(y, lam, rho, iters):
    """ADMM-based total variation denoising with improved solver"""
    # Convert to float64 and normalize
    y = y.astype(np.float64) / 255.0
    
    # Initialize variables
    x = y.copy()
    grad_x, grad_y = gradient(x)
    z1, z2 = grad_x.copy(), grad_y.copy()
    u1, u2 = np.zeros_like(z1, dtype=np.float64), np.zeros_like(z2, dtype=np.float64)
    
    # Store original for comparison
    x_prev = x.copy()
    
    for i in range(iters):
        # x-update: solve (I + rho*div*grad)x = y + rho*div(z-u)
        rhs = y + rho * divergence(z1 - u1, z2 - u2)
        x = solve_x_direct(rhs, rho, y.shape)
        
        # Ensure x stays in valid range
        x = np.clip(x, 0, 1)
        
        # z-update: soft thresholding with anisotropic TV
        grad_x, grad_y = gradient(x)
        z1 = shrink(grad_x + u1, lam / rho)
        z2 = shrink(grad_y + u2, lam / rho)
        
        # u-update: dual variable update
        u1 += grad_x - z1
        u2 += grad_y - z2
        
        # Check convergence
        if i % 10 == 0:
            primal_residual = np.sqrt(np.mean((grad_x - z1)**2 + (grad_y - z2)**2))
            dual_residual = np.sqrt(np.mean((x - x_prev)**2))
            
            if np.isnan(primal_residual) or np.isinf(primal_residual):
                print(f"  Warning: Numerical instability at iteration {i}")
                break
                
            print(f"  Iteration {i}, Primal: {primal_residual:.6f}, Dual: {dual_residual:.6f}")
            
            # Early stopping if converged
            if primal_residual < 1e-4 and dual_residual < 1e-4:
                print(f"  Converged at iteration {i}")
                break
        
        x_prev = x.copy()
    
    # Denormalize and clip
    result = (x * 255.0).clip(0, 255).astype(np.uint8)
    return result

# === PROCESS RGB IMAGES ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Could not load {filename}")
            continue

        print(f"Processing {filename}...")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split into R, G, B channels
        r, g, b = cv2.split(image_rgb)

        # Apply ADMM to each channel
        print("  Denoising R channel...")
        r_denoised = admm_denoise(r, lambda_tv, rho, iterations)
        print("  Denoising G channel...")
        g_denoised = admm_denoise(g, lambda_tv, rho, iterations)
        print("  Denoising B channel...")
        b_denoised = admm_denoise(b, lambda_tv, rho, iterations)

        # Merge denoised channels
        denoised_rgb = cv2.merge([r_denoised, g_denoised, b_denoised])

        # Convert back to BGR for saving
        denoised_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)

        # Save the result
        out_path = os.path.join(output_folder, f"denoised_{filename}")
        cv2.imwrite(out_path, denoised_bgr)
        print(f"  Saved: {out_path}")

print("🎯 All RGB images processed with ADMM denoising.")