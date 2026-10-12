"""经典机器学习 —— Phosphor 统计学习内核。

站在统计学习世界顶级成果之上（集大成）：
- PCA：对称协方差矩阵的特征分解（复用 linalg.jacobi_eigen），
  其秩-k 截断重建误差最优性由 Eckart-Young 定理（1936）保证——
  「任意秩-k 矩阵中，PCA 投影的 Frobenius 重建误差最小」。
- GMM：期望最大化算法（Dempster, Laird & Rubin 1977），
  对数似然单调不减是 EM 的定理性性质（不变量验证）。
- KMeans：Lloyd 算法（1957/1982），簇内平方和（惯性）单调不增。
- 高斯朴素贝叶斯：条件独立假设下的贝叶斯最优分类器。

不变量（tests 验证）：
- PCA：top-k 重建误差 ≤ bottom-k 重建误差（Eckart-Young）。
- GMM：每步对数似然 ≥ 上一步（容忍浮点抖动）。
- KMeans：每步惯性 ≤ 上一步。
- 责任度/权重行和为 1。
作者：晨星
"""

from __future__ import annotations

import numpy as np

from .linalg import jacobi_eigen, cholesky
from .numerical import logsumexp, softmax

__all__ = ["PCA", "GMM", "KMeans", "GaussianNB"]


# ---------------- PCA ----------------

class PCA:
    """主成分分析：协方差特征分解 + Eckart-Young 最优秩-k 重建。"""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # (k, d)
        self.eigvals_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs = jacobi_eigen(cov)  # 降序
        self.eigvals_ = eigvals
        k = self.n_components or d
        self.components_ = eigvecs[:, :k].T  # (k, d)
        total = np.sum(np.maximum(eigvals, 0))
        self.explained_variance_ratio_ = np.maximum(eigvals, 0)[:k] / total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        Z = self.transform(X)
        Xr = self.inverse_transform(Z)
        return float(np.mean(np.sum((X - Xr) ** 2, axis=1)))


# ---------------- GMM (EM) ----------------

class GMM:
    """高斯混合模型：期望最大化（EM）算法。"""

    def __init__(self, n_components: int = 2, max_iter: int = 200, tol: float = 1e-6,
                 seed: int = 0):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.weights_ = None
        self.means_ = None
        self.covs_ = None
        self.ll_history_ = []

    def _log_gaussian(self, X, mu, Sigma):
        d = X.shape[1]
        L = cholesky(Sigma)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        diff = X - mu
        sol = np.linalg.solve(L, diff.T).T  # (N, d)
        maha = np.sum(sol * sol, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        # 初始化：随机选 K 个点作均值，协方差取全局协方差
        idx = rng.choice(N, self.K, replace=False)
        self.means_ = X[idx].copy()
        self.covs_ = [np.cov(X.T) + 1e-6 * np.eye(d) for _ in range(self.K)]
        self.weights_ = np.full(self.K, 1.0 / self.K)
        resp = np.zeros((N, self.K))

        for it in range(self.max_iter):
            # E-step
            log_p = np.stack(
                [self._log_gaussian(X, self.means_[k], self.covs_[k]) for k in range(self.K)],
                axis=1,
            )  # (N, K)
            log_weighted = log_p + np.log(self.weights_ + 1e-300)
            ll = float(np.mean(logsumexp(log_weighted, axis=1)))
            self.ll_history_.append(ll)
            resp = softmax(log_weighted, axis=1)

            # M-step
            Nk = resp.sum(axis=0) + 1e-12
            self.weights_ = Nk / N
            self.means_ = (resp.T @ X) / Nk[:, None]
            for k in range(self.K):
                diff = X - self.means_[k]
                self.covs_[k] = (resp[:, k, None] * diff).T @ diff / Nk[k]
                self.covs_[k] += 1e-6 * np.eye(d)

            # 对数似然单调不减（EM 定理）
            if it > 0 and ll < self.ll_history_[-2] - 1e-6:
                # 允许极小浮点抖动，否则视为违反不变量
                break
            if it > 1 and abs(ll - self.ll_history_[-2]) < self.tol:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_p = np.stack(
            [self._log_gaussian(np.asarray(X, dtype=np.float64), self.means_[k], self.covs_[k])
             for k in range(self.K)],
            axis=1,
        )
        return np.argmax(log_p + np.log(self.weights_ + 1e-300), axis=1)

    def responsibilities(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        log_p = np.stack(
            [self._log_gaussian(X, self.means_[k], self.covs_[k]) for k in range(self.K)],
            axis=1,
        )
        return softmax(log_p + np.log(self.weights_ + 1e-300), axis=1)


# ---------------- KMeans ----------------

class KMeans:
    """KMeans：Lloyd 迭代算法，惯性（簇内平方和）单调不增。"""

    def __init__(self, n_clusters: int = 3, max_iter: int = 300, seed: int = 0):
        self.K = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.centroids_ = None
        self.labels_ = None
        self.inertia_history_ = []

    def _assign(self, X):
        # 向量化距离：||x-μ||² = ||x||² - 2 x·μ + ||μ||²
        dist2 = (np.sum(X * X, axis=1)[:, None]
                 - 2 * X @ self.centroids_.T
                 + np.sum(self.centroids_ * self.centroids_, axis=1)[None, :])
        labels = np.argmin(dist2, axis=1)
        inertia = float(np.sum(np.min(dist2, axis=1)))
        return labels, inertia

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        # k-means++ 初始化（Arthur & Vassilvitskii 2007，世界标准）
        centers = [X[rng.integers(N)]]
        for _ in range(1, self.K):
            dist2 = np.min(
                [(np.sum((X - c) ** 2, axis=1)) for c in centers], axis=0
            )
            probs = dist2 / dist2.sum()
            centers.append(X[rng.choice(N, p=probs)])
        self.centroids_ = np.array(centers, dtype=float)

        for _ in range(self.max_iter):
            labels, inertia = self._assign(X)
            self.inertia_history_.append(inertia)
            new_c = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids_[k]
                for k in range(self.K)
            ])
            if np.allclose(new_c, self.centroids_):
                break
            self.centroids_ = new_c
            labels, inertia = self._assign(X)
            self.inertia_history_.append(inertia)
        self.labels_ = labels
        return self


# ---------------- 高斯朴素贝叶斯 ----------------

class GaussianNB:
    """高斯朴素贝叶斯：条件独立假设下的贝叶斯最优分类器。"""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = None
        self.theta_ = None  # (n_classes, n_features) 均值
        self.sigma_ = None  # (n_classes, n_features) 方差

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_feat = X.shape[1]
        self.priors_ = np.zeros(len(self.classes_))
        self.theta_ = np.zeros((len(self.classes_), n_feat))
        self.sigma_ = np.zeros((len(self.classes_), n_feat))
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.priors_[i] = len(Xc) / len(X)
            self.theta_[i] = Xc.mean(axis=0)
            self.sigma_[i] = Xc.var(axis=0) + self.var_smoothing
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        # log N(x|μ,σ²) = -0.5[ln(2πσ²) + (x-μ)²/σ²]
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            terms = -0.5 * (np.log(2 * np.pi * self.sigma_[i])
                            + (X - self.theta_[i]) ** 2 / self.sigma_[i])
            jll[:, i] = np.sum(terms, axis=1) + np.log(self.priors_[i] + 1e-300)
        return jll

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        return softmax(jll, axis=1)
