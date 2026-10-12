"""自动微分 —— Phosphor 可微编程内核。

实现反向模式（reverse-mode）自动微分，对标 micrograd / 微型 autograd。
这是现代深度学习框架（PyTorch/TensorFlow/JAX）的数学内核：
- 计算图由前向传播隐式构建（tape-based）
- 反向传播用链式法则（链式法则 = 反向传播的母题）
- 支持标量 + 张量，覆盖 AI 系统全部基础算子

设计原则（集大成）：
- 不重造 CUDA/BLAS，张量运算复用 numpy；
- 梯度正确性由中心差分数值梯度交叉验证（见 nn.py / tests）。

不变量（tests 验证）：
- 任意复合函数 f，autodiff 梯度 == 中心差分数值梯度（相对误差 < 1e-5）
作者：晨星
"""

from __future__ import annotations

import numpy as np

__all__ = ["Var"]


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """把梯度缩减回参数原始形状：对广播出来的前置/单维求和。

    例如偏置 b 形状 (out,) 与批量激活 (N, out) 相加，反向时 b 的梯度
    必须沿批量维求和，得到 (out,) 而非 (N, out)。
    """
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Var:
    """可微变量：一次前向构造计算图，一次 backward 求全部梯度。"""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Var(shape={self.data.shape}, op={self._op!r})"

    # ---- 反向传播入口 ----
    def backward(self):
        topo: list["Var"] = []
        visited = set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        # 输出节点梯度初始化为 1（标量情形）；张量情形保持前导梯度形状
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # ---- 基础运算 ----
    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)

        out = Var(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)

        out = Var(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Var(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad = self.grad + (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1.0)

    def matmul(self, other):
        """矩阵乘（兼容 1-D/2-D）：dL/dA = dL/dY · Bᵀ，dL/dB = Aᵀ · dL/dY。"""
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data @ other.data, (self, other), "matmul")

        def _matmul_grads(gY, A, B):
            # 统一提升到 2-D 处理，结束再 squeeze 回原形状
            Ashape, Bshape = A.shape, B.shape
            if A.ndim == 1 and B.ndim == 1:  # 标量点积
                return gY * B, gY * A
            A2 = A[None, :] if A.ndim == 1 else A
            B2 = B[:, None] if B.ndim == 1 else B
            gY2 = gY[None, :] if (A.ndim == 1) else (gY[:, None] if B.ndim == 1 else gY)
            dA2 = gY2 @ B2.T
            dB2 = A2.T @ gY2
            dA = dA2.reshape(Ashape)
            dB = dB2.reshape(Bshape)
            return dA, dB

        def _backward():
            dA, dB = _matmul_grads(out.grad, self.data, other.data)
            self.grad = self.grad + dA
            other.grad = other.grad + dB

        out._backward = _backward
        return out

    def sum(self, axis=None):
        out = Var(np.sum(self.data, axis=axis), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad = self.grad + np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Var(np.mean(self.data, axis=axis), (self,), "mean")

        def _backward():
            grad = out.grad / n
            if axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad = self.grad + np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def relu(self):
        out = Var(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Var(s, (self,), "sigmoid")

        def _backward():
            self.grad = self.grad + s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Var(t, (self,), "tanh")

        def _backward():
            self.grad = self.grad + (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Var(np.log(self.data), (self,), "log")

        def _backward():
            self.grad = self.grad + (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Var(e, (self,), "exp")

        def _backward():
            self.grad = self.grad + e * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1):
        """对数域稳定的 softmax（复用 numerical 技巧），可直接求交叉熵梯度。"""
        from .numerical import softmax as _softmax

        out = Var(_softmax(self.data, axis=axis), (self,), "softmax")

        def _backward():
            s = out.data
            # Jacobian-Vector Product：J = diag(s) - s sᵀ
            g = out.grad
            self.grad = self.grad + (g - np.sum(g * s, axis=axis, keepdims=True)) * s

        out._backward = _backward
        return out
