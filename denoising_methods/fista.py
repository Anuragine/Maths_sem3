import os
import cv2
import numpy as np

# === PARAMETERS ===
lambda_tv = 0.05     # regularization strength
step_size = 0.25     # step size (should be < 1/L where L is Lipschitz constant)
iterations = 10
     # number of FISTA iterations

input_folder = 'input_imgs'
output_folder = 'denoised_fista'
os.makedirs(output_folder, exist_ok=True)

# === HELPER FUNCTIONS ===
def gradient(img):
    """Forward difference gradients"""
    grad_x = np.zeros_like(img, dtype=np.float64)
    grad_y = np.zeros_like(img, dtype=np.float64)
    grad_x[:, :-1] = img[:, 1:] - img[:, :-1]
    grad_y[:-1, :] = img[1:, :] - img[:-1]
    return grad_x, grad_y

def divergence(grad_x, grad_y):
    """Adjoint of gradient operator"""
    div = np.zeros_like(grad_x, dtype=np.float64)
    div[:, 0] = grad_x[:, 0]
    div[:, 1:-1] = grad_x[:, 1:-1] - grad_x[:, :-2]
    div[:, -1] = -grad_x[:, -2]
    temp = np.zeros_like(grad_y, dtype=np.float64)
    temp[0, :] = grad_y[0, :]
    temp[1:-1, :] = grad_y[1:-1] - grad_y[:-2, :]
    temp[-1, :] = -grad_y[-2, :]
    div += temp
    return div

# === FISTA DENOISING FUNCTION ===
def fista_denoise(y, lam, step, iters):
    """FISTA-based total variation denoising"""
    y = y.astype(np.float64) / 255.0
    x = y.copy()
    z = x.copy()  # momentum variable
    t = 1.0       # acceleration term
    
    for i in range(iters):
        # Gradient of data fidelity: (z - y)
        grad_f = z - y
        
        # Gradient step
        x_temp = z - step * grad_f
        
        # TV shrinkage
        grad_x, grad_y = gradient(x_temp)
        mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
        shrink_factor = np.maximum(1 - (lam * step) / mag, 0)
        grad_x = grad_x * shrink_factor
        grad_y = grad_y * shrink_factor
        
        # Update x
        x_new = x_temp + step * divergence(grad_x, grad_y)
        x_new = np.clip(x_new, 0, 1)
        
        # Momentum update
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        
        # Update variables
        x, t = x_new, t_new
        
        if i % 20 == 0:
            print(f"Iteration {i}/{iters}")
    
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

        print(f"Processing {filename} with FISTA...")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split channels
        r, g, b = cv2.split(image_rgb)

        # Apply FISTA to each channel
        r_denoised = fista_denoise(r, lambda_tv, step_size, iterations)
        g_denoised = fista_denoise(g, lambda_tv, step_size, iterations)
        b_denoised = fista_denoise(b, lambda_tv, step_size, iterations)

        # Merge channels
        denoised_rgb = cv2.merge([r_denoised, g_denoised, b_denoised])

        # Convert back to BGR for saving
        denoised_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)

        # Save
        out_path = os.path.join(output_folder, f"denoised_fista_{filename}")
        cv2.imwrite(out_path, denoised_bgr)
        print(f"  Saved: {out_path}")

print("🎯 All images processed with FISTA denoising.")
