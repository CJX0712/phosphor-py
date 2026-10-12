"""线性代数 —— Phosphor 代数内核。

站在世界顶级算法之上（集大成）：
- Jacobi 循环旋转法（Jacobi 1846）做对称阵特征分解：数值最稳定，
  特征值/特征向量同时收敛，特征向量天然正交，是 SVD/PCA 的黄金算法。
- 幂迭代（Power Iteration）+ 逐一降维（deflation）求前 k 主特征对。
- Cholesky 分解（正定阵平方根），带 jitter 兜底保证半正定可分解。

不变量（tests 验证）：
- ‖A v_i - λ_i v_i‖ ≈ 0（特征方程残差）
- Vᵀ V = I（特征向量正交归一）
- Cholesky: L Lᵀ ≈ A
作者：晨星
"""

from __future__ import annotations

import numpy as np

__all__ = ["jacobi_eigen", "power_iteration", "top_eigen", "cholesky", "matmul"]


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """矩阵乘法封装（复用 numpy，世界标准 BLAS）。"""
    return np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def jacobi_eigen(A: np.ndarray, max_sweeps: int = 100, tol: float = 1e-13):
    """经典循环 Jacobi 特征分解（对称阵）。

    参数
    ----
    A : (n, n) 对称矩阵
    返回
    ----
    eigvals : (n,) 特征值（降序）
    eigvecs : (n, n) 列向量为对应特征向量，正交归一

    算法：反复寻找最大非对角元 (p,q)，构造 Givens 旋转使其归零。
    每轮 sweep 遍历所有上三角对；收敛后非对角 Frobenius 范数→0。
    """
    A = np.asarray(A, dtype=np.float64).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("jacobi_eigen 需要方阵")
    n = A.shape[0]
    V = np.eye(n)

    for _ in range(max_sweeps):
        # 非对角 Frobenius 范数
        off = np.sqrt(2.0 * np.sum(np.triu(A, k=1) ** 2))
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p, q]) < 1e-300:
                    continue
                # 计算旋转角 θ → t = tan(φ)
                theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                t = np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                # 旋转：A ← Jᵀ A J
                App, Aqq = A[p, p], A[q, q]
                A[p, p] = c * c * App - 2 * s * c * A[p, q] + s * s * Aqq
                A[q, q] = s * s * App + 2 * s * c * A[p, q] + c * c * Aqq
                A[p, q] = 0.0
                A[q, p] = 0.0
                for r in range(n):
                    if r == p or r == q:
                        continue
                    Arp, Arq = A[r, p], A[r, q]
                    A[r, p] = A[p, r] = c * Arp - s * Arq
                    A[r, q] = A[q, r] = s * Arp + c * Arq
                # 累积旋转到特征向量矩阵
                for r in range(n):
                    Vrp, Vrq = V[r, p], V[r, q]
                    V[r, p] = c * Vrp - s * Vrq
                    V[r, q] = s * Vrp + c * Vrq

    eigvals = np.diag(A)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], V[:, order]


def power_iteration(A: np.ndarray, max_iter: int = 10000, tol: float = 1e-14):
    """幂迭代求主特征对（dominant eigenpair）。

    v_{k+1} = A v_k / ‖A v_k‖，收敛到模最大特征值对应的特征向量。
    带 Rayleigh 商估计特征值，残差 ‖A v - λ v‖ 收敛到 0。
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(max_iter):
        w = A @ v
        v = w / np.linalg.norm(w)
        lam = float(v @ (A @ v))  # Rayleigh 商
        if np.linalg.norm(A @ v - lam * v) < tol:
            break
    return lam, v


def top_eigen(A: np.ndarray, k: int, max_iter: int = 10000, tol: float = 1e-12):
    """前 k 个主特征对：幂迭代 + 隐式 deflation（Hotelling 1929）。

    每求得一个主特征对，用 A' = A - λ v vᵀ 移除其分量，再迭代下一对。
    返回 (eigvals[k], eigvecs[n,k])，降序。
    """
    A = np.asarray(A, dtype=np.float64).copy()
    n = A.shape[0]
    k = min(k, n)
    vals = np.zeros(k)
    vecs = np.zeros((n, k))
    for i in range(k):
        lam, v = power_iteration(A, max_iter=max_iter, tol=tol)
        vals[i] = lam
        vecs[:, i] = v
        # deflation：移除已求得的特征方向
        A = A - lam * np.outer(v, v)
    return vals, vecs


def cholesky(A: np.ndarray, jitter: float = 1e-8):
    """Cholesky 分解 A = L Lᵀ（下三角 L），带 jitter 兜底半正定。

    若矩阵非严格正定，自动加 jitter*I 使其可分解，保证数值可用
    （GMM/卡尔曼滤波等协方差分解的工业标准做法）。
    """
    A = np.asarray(A, dtype=np.float64).copy()
    n = A.shape[0]
    jitter_i = 0.0
    while True:
        try:
            L = np.linalg.cholesky(A + jitter_i * np.eye(n))
            return L
        except np.linalg.LinAlgError:
            jitter_i = jitter if jitter_i == 0 else jitter_i * 10
            if jitter_i > 1e3:
                raise
