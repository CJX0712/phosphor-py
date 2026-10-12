"""数值稳定原语 —— Phosphor 数值地基。

设计哲学（集大成）：
- 不重造 BLAS/LAPACK，底层复用 numpy（世界标准数值计算）。
- 但凡涉及指数/对数/概率，一律采用世界顶级数值稳定写法：
  * logsumexp 防溢出（机器学习 / 统计推断的基石，见 Bishop PRML §4.4）
  * 减去最大值的稳定 softmax（避免 exp 上溢）
  * softplus / log-sigmoid 的稳定形式
- 提供中心差分数值梯度，作为「不变量」交叉验证的黄金参照。

所有函数均通过 tests/test_all.py 中的不变量检验。
作者：晨星
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "logsumexp",
    "softmax",
    "log_softmax",
    "softplus",
    "log_sigmoid",
    "stable_sigmoid",
    "numerical_gradient",
    "numerical_jacobian",
    "clip_grad_norm",
]


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """数值稳定的 log(sum(exp(a)))。

    技巧：提取每行最大值 m，计算 m + log(sum(exp(a - m)))，
    将指数项约束在 [exp(-∞), 1] 区间，彻底消除上溢。
    这是变分推断、softmax、CRF 等一切对数域运算的通用 primitive。
    """
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a, axis=axis, keepdims=True)
    # 处理全 -inf 的退化情形，避免 nan
    amax = np.where(np.isfinite(amax), amax, 0.0)
    s = np.sum(np.exp(a - amax), axis=axis, keepdims=True)
    out = amax + np.log(s)
    if axis is None:
        return out.reshape(())  # 标量
    return np.squeeze(out, axis=axis) if s.shape[axis] == 1 else out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax。

    减去最大值后归一化，保证：(1) 不溢出；(2) 输出每行（沿 axis）和为 1。
    不变量：out.sum(axis) == 1（在浮点容差内）。
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(x))，直接由 logsumexp 计算，双重稳定。"""
    x = np.asarray(x, dtype=np.float64)
    return x - logsumexp(x, axis=axis)


def softplus(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softplus: log(1 + exp(x))。

    大正 x 时退化为 x；大负 x 时用 log1p(exp(x)) 避免 exp 上溢。
    """
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 30.0, x, np.log1p(np.exp(x)))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 log(sigmoid(x)) = -softplus(-x)。"""
    return -softplus(-np.asarray(x, dtype=np.float64))


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，左右两侧均不溢出。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心差分数值梯度 —— 解析/自动微分的黄金参照。

    对每个分量 i：df/dx_i ≈ (f(x+εe_i) - f(x-εe_i)) / (2ε)。
    中心差分误差阶 O(ε²)，比前向差分 O(ε) 更精确。
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = float(np.asarray(f(x)).item() if np.ndim(f(x)) == 0 else np.sum(f(x)))
        x[idx] = orig - eps
        fm = float(np.asarray(f(x)).item() if np.ndim(f(x)) == 0 else np.sum(f(x)))
        x[idx] = orig
        grad[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心差分雅可比。f: R^n -> R^m，返回 m×n 雅可比矩阵。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    f0 = np.asarray(f(x), dtype=np.float64).ravel()
    m = f0.size
    J = np.zeros((m, n))
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        fp = np.asarray(f(xp), dtype=np.float64).ravel()
        fm = np.asarray(f(xm), dtype=np.float64).ravel()
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J


def clip_grad_norm(grads: list[np.ndarray], max_norm: float) -> float:
    """梯度范数裁剪（Transformer 训练世界标准技巧，Pascanu et al. 2013）。

    返回裁剪前的全局范数，并将 grads 就地缩放到 max_norm 内。
    """
    total = np.sqrt(sum(np.sum(g * g) for g in grads))
    if total > max_norm > 0:
        scale = max_norm / (total + 1e-12)
        for g in grads:
            g *= scale
    return float(total)
