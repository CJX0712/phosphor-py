"""注意力机制 —— Phosphor Transformer 内核。

站在 Transformer 原始论文之上（Vaswani et al. 2017, "Attention Is All You Need"）：
    Attention(Q,K,V) = softmax(Q Kᵀ / √d_k) V

集大成：
- 缩放因子 1/√d_k 防止点积过大压入 softmax 饱和区（原论文关键设计）。
- 数值稳定 softmax（减去最大值，见 numerical.py）——与主流框架一致。
- 多头注意力：并行 h 个子空间，拼接后经输出投影（原论文 §3.2.2）。
- 因果掩码（causal mask）：解码器自注意力的下三角可见性。

不变量（tests 验证）：
- 注意力权重每行（沿 key 维度）和为 1。
- 因果掩码下，位置 i 仅关注 ≤ i 的位置，其余权重严格为 0。
- 多头实现与按公式手算的单头展开结果一致（容差 1e-9）。
作者：晨星
"""

from __future__ import annotations

import numpy as np

from .numerical import softmax

__all__ = ["scaled_dot_product_attention", "MultiHeadAttention"]


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    causal: bool = False,
):
    """缩放点积注意力。

    参数
    ----
    Q : (..., seq_q, d_k)
    K : (..., seq_k, d_k)
    V : (..., seq_k, d_v)
    mask : 可选 (..., seq_q, seq_k) 布尔/浮点掩码，True/1 表示屏蔽
    causal : 若为 True，施加下三角因果掩码（解码器自注意力）

    返回
    ----
    out : (..., seq_q, d_v)
    attn : (..., seq_q, seq_k) 注意力权重（每行和为 1）
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_k = Q.shape[-1]

    #  scores = Q Kᵀ / √d_k
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        causal_mask = np.tril(np.ones((seq_q, seq_k), dtype=bool))
        # 在最后两维广播
        while causal_mask.ndim < scores.ndim:
            causal_mask = causal_mask[None]
        scores = np.where(causal_mask, scores, -np.inf)

    if mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool), -np.inf, scores)

    attn = softmax(scores, axis=-1)
    out = np.matmul(attn, V)
    return out, attn


class MultiHeadAttention:
    """多头注意力（原论文 §3.2.2），参数随机初始化。

    head_i(Q,K,V) = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    MultiHead = Concat(head_1..h) W^O
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = np.sqrt(2.0 / d_model)  # Xavier 风格
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale

    def forward(self, X: np.ndarray, causal: bool = False):
        X = np.asarray(X, dtype=np.float64)
        N, T, _ = X.shape
        h = self.n_heads
        dk = self.d_k

        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        # 拆头：(N, T, h, dk) -> (N, h, T, dk)
        Qh = Q.reshape(N, T, h, dk).transpose(0, 2, 1, 3)
        Kh = K.reshape(N, T, h, dk).transpose(0, 2, 1, 3)
        Vh = V.reshape(N, T, h, dk).transpose(0, 2, 1, 3)

        # 各头独立注意力
        attn_heads = []
        for i in range(h):
            out_i, _ = scaled_dot_product_attention(
                Qh[:, i], Kh[:, i], Vh[:, i], causal=causal
            )
            attn_heads.append(out_i)
        # 拼回 (N, h, T, dk) -> (N, T, h, dk) -> (N, T, d_model)
        concat = np.stack(attn_heads, axis=1).transpose(0, 2, 1, 3).reshape(N, T, h * dk)
        out = concat @ self.Wo
        return out

    def params(self) -> list[np.ndarray]:
        return [self.Wq, self.Wk, self.Wv, self.Wo]
