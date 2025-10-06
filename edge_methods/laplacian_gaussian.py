import cv2
import numpy as np

kernel = np.array([[0,0, -1, 0,0],
                   [0,-1, -2, -1,0],
                   [-1,-2,16,-2, -1],
                   [0,-1, -2, -1,0],
                   [0,0, -1, 0,0]], dtype=np.float32)  # Sharpen kernel

# Read image (grayscale for edge detection/sharpening)
img = cv2.imread("denoised_ista\denoised_ista_000001.jpeg", cv2.IMREAD_GRAYSCALE)

# Convolve with kernel
filtered_img = cv2.filter2D(img, -1, kernel)

# Save result
cv2.imwrite("laplcian_gaussian.png", filtered_img)
print("✅ Output saved as output.png")