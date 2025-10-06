import cv2
import numpy as np

# ✅ Use forward slashes for Linux paths
img = cv2.imread("input_imgs/000001.jpeg", cv2.IMREAD_GRAYSCALE)

# Kernel 
kernel_x = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype=np.float32)

kernel_y = np.array([[ 1,  1,  1],
                     [ 0,  0,  0],
                     [-1, -1, -1]], dtype=np.float32)

# Apply Prewitt filters
Gx = cv2.filter2D(img, cv2.CV_64F, kernel_x)
Gy = cv2.filter2D(img, cv2.CV_64F, kernel_y)

# Hybrid magnitude
hybrid = np.sqrt(Gx**2 + Gy**2)
hybrid = cv2.convertScaleAbs(hybrid)

cv2.imwrite("edge_images/prewitt_hybrid.png", hybrid)
print("✅ Output saved as Maths_sem3/edge_images/prewitt_hybrid.png")
