"""
README (project summary and usage)

This script evaluates image processing results (denoising and edge detection)
by computing PSNR, SSIM and MSE against a reference image. It iterates over
several result folders, writes per-image metrics and averages (for denoising
methods) into `evaluation_results.csv`, and prints folder summaries.

Project structure (relevant paths):
- input_imgs/: contains reference/source images (e.g. 000001.jpeg)
- denoised_ista/, denoised_admm/, denoised_fista/: denoised output images
- edge_images/: edge-detected outputs (evaluated but not averaged)
- evaluation_results.csv: output CSV with per-image and average metrics

Usage:
1. Ensure required Python packages are installed (see Dependencies).
2. Run this script from the repository root:
     python Evaluation/Main.py
3. Inspect `evaluation_results.csv` for per-image metrics and averages.

Notes and assumptions:
- The script expects grayscale images; it reads images with cv2.IMREAD_GRAYSCALE.
- If result images differ in size from the reference, they will be resized to
    match the reference dimensions before metric computation.
- Edge images are evaluated and written to CSV but are excluded from folder
    averaging because they are not direct denoising outputs.
- The default reference image is `input_imgs/000001.jpeg`. Change the
    `reference_path` variable in the script to evaluate a different reference.

Output CSV columns: Method, Image_Name, PSNR, SSIM, MSE

"""

import os
import cv2
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Folders
methods = ["denoised_ista", "denoised_admm", "denoised_fista", "edge_images"]

# Reference image
reference_path = "input_imgs/000001.jpeg"
reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

# Functions to compute metrics (all built in)
def compute_mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=255)

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=255)

# CSV file to save results
csv_file = "evaluation_results.csv"
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Image_Name", "PSNR", "SSIM", "MSE"])

    # Evaluation
    for method in methods:
        folder_path = method
        print(f"\n--- Evaluating {method} ---")

        total_psnr = 0
        total_ssim = 0
        total_mse = 0
        count = 0

        for file in os.listdir(folder_path):
            if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Resize if necessary
                if reference.shape != img.shape:
                    img = cv2.resize(img, (reference.shape[1], reference.shape[0]))

                # Compute metrics
                psnr_val = compute_psnr(reference, img)
                ssim_val = compute_ssim(reference, img)
                mse_val = compute_mse(reference, img)

                # Write individual image result to CSV
                writer.writerow([method, file, f"{psnr_val:.4f}", f"{ssim_val:.4f}", f"{mse_val:.4f}"])

                # For averaging only if not edge_images
                if method != "edge_images":
                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    total_mse += mse_val
                    count += 1

        # Write average metrics for folders except edge_images
        if method != "edge_images" and count > 0:
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            avg_mse = total_mse / count
            writer.writerow([method, "Average", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", f"{avg_mse:.4f}"])
            print(f"Average PSNR : {avg_psnr:.4f}")
            print(f"Average SSIM : {avg_ssim:.4f}")
            print(f"Average MSE  : {avg_mse:.4f}")
        elif method == "edge_images":
            print("Edge images evaluated")
