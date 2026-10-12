"""Phosphor —— 世界顶级 AI 数学与算法内核。

架构（从数学地基到 AI 模型，十层自底向上）：
  1. numerical  —— 数值稳定原语（logsumexp / 稳定 softmax / 数值梯度）
  2. linalg     —— 线性代数（Jacobi 特征分解 / 幂迭代 / Cholesky）
  3. autodiff   —— 反向模式自动微分（可微编程内核）
  4. nn         —— 神经网络（MLP / Adam / 损失函数）
  5. attention  —— 注意力机制（Transformer 内核）
  6. ml         —— 经典机器学习（PCA / GMM-EM / KMeans / 高斯朴素贝叶斯）

核心原则：
- 零重写：底层复用 numpy（世界标准数值计算），不重造 BLAS/LAPACK。
- 集大成：每个算法站在世界顶级成果之上（Kingma&Ba / Vaswani / Eckart-Young / Jacobi ...）。
- 闭环交付：每条算法带「不变量」自检，解析解 vs 数值解 vs 自动微分三重交叉验证。
作者：晨星
"""

from __future__ import annotations

from . import numerical, linalg, autodiff, nn, attention, ml
from .numerical import (
    logsumexp, softmax, log_softmax, softplus, log_sigmoid,
    stable_sigmoid, numerical_gradient, numerical_jacobian, clip_grad_norm,
)
from .linalg import jacobi_eigen, power_iteration, top_eigen, cholesky, matmul
from .autodiff import Var
from .nn import (
    Linear, MLP, AutodiffMLP, SGD, Adam, mse_loss, cross_entropy_loss,
)
from .attention import scaled_dot_product_attention, MultiHeadAttention
from .ml import PCA, GMM, KMeans, GaussianNB

__version__ = "1.0.0"
__author__ = "晨星"

__all__ = [
    "numerical", "linalg", "autodiff", "nn", "attention", "ml",
    "logsumexp", "softmax", "log_softmax", "softplus", "log_sigmoid",
    "stable_sigmoid", "numerical_gradient", "numerical_jacobian", "clip_grad_norm",
    "jacobi_eigen", "power_iteration", "top_eigen", "cholesky", "matmul",
    "Var",
    "Linear", "MLP", "AutodiffMLP", "SGD", "Adam", "mse_loss", "cross_entropy_loss",
    "scaled_dot_product_attention", "MultiHeadAttention",
    "PCA", "GMM", "KMeans", "GaussianNB",
]
