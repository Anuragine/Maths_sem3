
# 🧮 Denoising and Edge Preservation in Medical Images

## 📌 Project Overview

Medical images such as CT and MRI scans often suffer from **noise** and **blur**, which can reduce diagnostic accuracy.
This project explores **mathematical optimization methods** for denoising and **classical convolution kernels** for edge preservation.

The workflow:

1. **Denoising** (remove noise while retaining structures)

   * Constrained optimization techniques (ISTA, FISTA, ADMM)
   * Wavelet thresholding
2. **Edge detection & sharpening**

   * Classical kernels (Sobel, Prewitt, Scharr, Laplacian, Roberts, Gaussian, Bilaplacian, etc.)
   * Edge-preserving sharpening filters
3. **Comparison of results**

   * Effectiveness of different methods on noisy medical images

---

## 🔬 Mathematical Background

### 1. Image Denoising as Optimization

The noisy image can be modeled as:

$$
y = x + n
$$

where:

* ( y ) = observed noisy image
* ( x ) = true clean image
* ( n ) = noise

We want to recover ( x ) from ( y ).

This can be framed as an optimization problem:

$$
\min_x ; \frac{1}{2} |y - x|_2^2 + \lambda R(x)
$$

* First term = **data fidelity** (keep close to observed image)
* Second term = **regularization** (penalize noise, enforce smoothness or sparsity)
* ( R(x) ) can be Total Variation (TV), or ( L_1 )-norm in wavelet domain.

---

### 2. Iterative Shrinkage-Thresholding Algorithm (ISTA)

ISTA solves the optimization via iterative updates:

$$
x^{k+1} = S_{\lambda/L} \Big( x^k - \frac{1}{L} \nabla f(x^k) \Big)
$$

where:

* ( S_{\theta}(z) = \text{sign}(z) \cdot \max(|z| - \theta, 0) ) (**soft-thresholding**)
* ( L ) = Lipschitz constant of gradient

---

### 3. Fast ISTA (FISTA)

FISTA accelerates ISTA using momentum:

$$
\begin{aligned}
y^k &= x^k + \frac{t_{k-1}-1}{t_k} (x^k - x^{k-1}) \
x^{k+1} &= S_{\lambda/L} \Big( y^k - \frac{1}{L} \nabla f(y^k) \Big)
\end{aligned}
$$

This converges much faster than ISTA.

---

### 4. Alternating Direction Method of Multipliers (ADMM)

ADMM solves constrained problems of the form:

$$
\min_{x,z} ; f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c
$$

**Augmented Lagrangian:**

$$
L(x,z,u) = f(x) + g(z) + u^T(Ax+Bz-c) + \frac{\rho}{2}|Ax+Bz-c|_2^2
$$

Update steps alternate between ( x ), ( z ), and dual variable ( u ).
This is widely used in **Total Variation (TV) denoising**.

---

### 5. Wavelet Thresholding

Wavelet transform separates image into frequency bands:

* Noise → high-frequency
* Structure → low-frequency

Thresholding rule:

**Hard thresholding:**

$$
w' =
\begin{cases}
w & |w| \geq \lambda \
0 & |w| < \lambda
\end{cases}
$$

**Soft thresholding:**

$$
w' = \text{sign}(w) \cdot \max(|w| - \lambda, 0)
$$

Reconstruct using inverse wavelet transform to get denoised image.

---

## ⚙️ Edge Detection and Sharpening Kernels

After denoising, edges must be preserved. Convolution kernels are applied to highlight boundaries.

---

### 1. Sobel Operator

Emphasizes horizontal/vertical edges.

$$
K_x =
\begin{bmatrix}
-1 & 0 & 1 \
-2 & 0 & 2 \
-1 & 0 & 1
\end{bmatrix},
\quad
K_y =
\begin{bmatrix}
-1 & -2 & -1 \
0 & 0 & 0 \
1 & 2 & 1
\end{bmatrix}
$$

---

### 2. Prewitt Operator

Similar to Sobel, but equal weights.

$$
K_x =
\begin{bmatrix}
-1 & 0 & 1 \
-1 & 0 & 1 \
-1 & 0 & 1
\end{bmatrix},
\quad
K_y =
\begin{bmatrix}
1 & 1 & 1 \
0 & 0 & 0 \
-1 & -1 & -1
\end{bmatrix}
$$

---

### 3. Roberts Cross Operator

Detects diagonal edges.

$$
K_x =
\begin{bmatrix}
1 & 0 \
0 & -1
\end{bmatrix},
\quad
K_y =
\begin{bmatrix}
0 & 1 \
-1 & 0
\end{bmatrix}
$$

---

### 4. Scharr Operator

Improved rotational symmetry compared to Sobel.

$$
K_x =
\begin{bmatrix}
-3 & 0 & 3 \
-10 & 0 & 10 \
-3 & 0 & 3
\end{bmatrix},
\quad
K_y =
\begin{bmatrix}
-3 & -10 & -3 \
0 & 0 & 0 \
3 & 10 & 3
\end{bmatrix}
$$

---

### 5. Laplacian Operator

Second derivative, isotropic edge detection.

$$
K =
\begin{bmatrix}
0 & -1 & 0 \
-1 & 4 & -1 \
0 & -1 & 0
\end{bmatrix}
$$

Variants:

* `laplacian_1.py` (4-neighborhood)
* `laplacian_2.py` (8-neighborhood)
* `laplacian_gaussian.py` (pre-smoothed Laplacian to reduce noise sensitivity)

---

### 6. Gaussian Blur Kernel

Smooths image before edge detection.

$$
G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
$$

---

### 7. Bilaplacian

Higher-order Laplacian operator. Useful for strong edge emphasis.

---

### 8. Bilateral Filter

Edge-preserving smoothing.

$$
I'(x) = \frac{1}{W_p} \sum_{y \in \Omega} I(y) \cdot f_s(|x-y|) \cdot f_r(|I(x)-I(y)|)
$$

where:

* ( f_s ) = spatial closeness
* ( f_r ) = intensity similarity

---

### 9. General Sharpening Kernel

Enhances edges:

$$
K =
\begin{bmatrix}
0 & -1 & 0 \
-1 & 5 & -1 \
0 & -1 & 0
\end{bmatrix}
$$

---

## 🖥️ How to Run

1. Clone repo:

```bash
git clone https://github.com/yourusername/denoising-edge-preservation.git
cd denoising-edge-preservation
```

2. Place your input image in the project folder.
3. Run any script:

```bash
python sobel_horizontal_edge.py
```

4. Outputs will be saved as images (e.g., `output.png`).

---

## 📊 Observations

* **Gaussian Blur**: Removes noise but also blurs fine details.
* **Bilateral Filter**: Removes noise while preserving edges.
* **Sobel/Prewitt/Scharr**: Detect directional edges (x, y).
* **Roberts Cross**: Detects diagonal edges.
* **Laplacian**: Detects edges in all directions but more noise-sensitive.
* **Sharpening Kernels**: Enhance edges but can amplify noise.
* **ISTA/FISTA/ADMM**: Provide optimization-based denoising, less prone to over-smoothing than Gaussian blur.

---

## 🚀 Future Work

* Implement **hybrid pipelines** (e.g., Bilateral + Laplacian).
* Explore **non-convolution ML/DL approaches** (SVM edge classifiers, CNN-based edge detectors).
* Compare runtime and accuracy across all methods.

