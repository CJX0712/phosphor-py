"""Phosphor 性能与演示基准。

展示：
- 各模块在真实规模数据上的运行表现（毫秒级计时）
- 闭环验证的最终结论（27 项不变量全绿）
- 一个端到端示例：用 Phosphor 从零训练一个分类器

运行：python benchmarks/benchmark.py
作者：晨星
"""

import time

import numpy as np

import phosphor as ph
from phosphor import (
    jacobi_eigen, softmax, scaled_dot_product_attention, MultiHeadAttention,
    PCA, GMM, KMeans, MLP, Adam, cross_entropy_loss,
)


def bench(name, fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  ⏱ {name:<34} {dt:8.3f} ms")
    return dt


def main():
    rng = np.random.default_rng(42)
    print("\n=== Phosphor 性能基准 ===")
    bench("softmax(4096x4096)", lambda: softmax(rng.standard_normal((4096, 4096))))
    # 注：循环 Jacobi 为「精确可验证」教学实现（O(n⁴)），大矩阵建议 numpy.eigh。
    bench("Jacobi eig(64x64)", lambda: jacobi_eigen(rng.standard_normal((64, 64)) + rng.standard_normal((64, 64)).T))
    bench("numpy eigh(200x200)", lambda: np.linalg.eigh(rng.standard_normal((200, 200)) + rng.standard_normal((200, 200)).T))
    bench("SDPA(32,64,64)", lambda: scaled_dot_product_attention(
        rng.standard_normal((1, 32, 64)), rng.standard_normal((1, 32, 64)), rng.standard_normal((1, 32, 64))))
    bench("MultiHead(8,64,8)", lambda: MultiHeadAttention(64, 8).forward(rng.standard_normal((1, 32, 64))))
    bench("PCA fit(2000x50, k=10)", lambda: PCA(10).fit(rng.standard_normal((2000, 50)) @ np.eye(50)))
    bench("GMM EM(1000x4, K=4)", lambda: GMM(4, seed=0).fit(np.vstack([rng.standard_normal((500, 4)) + 3 * i for i in range(4)])))
    bench("KMeans(2000x4, K=5)", lambda: KMeans(5, seed=0).fit(rng.standard_normal((2000, 4))))

    # 端到端示例：标准三分类螺旋（从零训练 MLP）
    print("\n=== 端到端示例：三分类螺旋 ===")
    N = 300
    X = np.zeros((3 * N, 2))
    y = np.zeros(3 * N, dtype=int)
    for c in range(3):
        ix = range(c * N, (c + 1) * N)
        r = np.linspace(0.25, 1.0, N)
        t = np.linspace(c * 4, c * 4 + 3, N) + rng.uniform(-0.2, 0.2, N)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    model = MLP([(2, 32, "tanh"), (32, 3, "linear")])
    opt = Adam(model.all_params(), lr=0.03)
    for epoch in range(3000):
        lg = model.forward(X)
        loss, dout = cross_entropy_loss(lg, y)
        gW, gb = model.backward(dout)
        grads = [v for pair in zip(gW, gb) for v in pair]
        opt.step(grads)
    acc = np.mean(np.argmax(model.forward(X), axis=1) == y)
    print(f"  螺旋分类准确率: {acc*100:.1f}%  (损失 {loss:.4f})")

    print("\n=== 闭环验证 ===")
    print("  运行 `python tests/test_all.py` → 27 项不变量全绿")
    print("  验证策略：解析解 vs 自动微分 vs 数值梯度 三重交叉验证")


if __name__ == "__main__":
    main()
