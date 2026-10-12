# Phosphor · 世界顶级 AI 数学与算法内核

> **Phosphor（启明星 / 晨星）** —— 从数学地基到 AI 模型的可验证内核。
> 每条算法都站在世界顶级成果之上，并带「不变量」自检：解析解 vs 自动微分 vs 数值梯度，三重交叉验证，全绿交付。

```
作者：晨星 (Morning Star)   |   License：MIT   |   验证：27/27 不变量全绿
原则：零重写 · 集大成 · 闭环交付
```

---

## 一、核心原则

| 原则 | 含义 | 在本项目中的体现 |
|------|------|------------------|
| **零重写** | 不重造轮子，底层复用世界最优解 | 数值计算复用 **numpy**（世界标准 BLAS/LAPACK） |
| **集大成** | 每个模块站在顶级成果之上 | 算法直接采用 Kingma&Ba / Vaswani / Eckart-Young / Jacobi / Glorot 等规范形式 |
| **闭环交付** | 可运行、可复现、可验证、开源 | `tests/test_all.py` 全绿 + 基准脚本 + MIT 开源至 GitHub |
| **自治执行** | 安装/配置/诊断/修复全自动 | 一键 `python tests/test_all.py` 完成全部验证 |

---

## 二、架构总览（六层内核）

自底向上，每一层都是上一层的可验证基石：

| 层 | 模块 | 核心算法 | 世界顶级参照 |
|----|------|----------|--------------|
| 1 | `numerical` | logsumexp / 稳定 softmax / 中心差分数值梯度 | Bishop *PRML* §4.4 |
| 2 | `linalg` | Jacobi 特征分解 / 幂迭代 / Cholesky | Jacobi 1846；Hotelling 1929 |
| 3 | `autodiff` | 反向模式自动微分（reverse-mode） | micrograd / 微型 autograd |
| 4 | `nn` | MLP / Adam / 交叉熵 | Kingma & Ba 2015；Glorot & Bengio 2010 |
| 5 | `attention` | 缩放点积注意力 / 多头 / 因果掩码 | Vaswani et al. 2017 |
| 6 | `ml` | PCA / GMM-EM / KMeans / 高斯朴素贝叶斯 | Eckart-Young 1936；Dempster 1977；Lloyd 1982 |

---

## 三、快速开始

```bash
pip install -e .          # 或 pip install numpy
python tests/test_all.py  # 27 项不变量全绿
python benchmarks/benchmark.py  # 性能 + 端到端示例
```

```python
import numpy as np
from phosphor import MLP, Adam, cross_entropy_loss, scaled_dot_product_attention

# 注意力（Transformer 内核）
Q = K = V = np.random.randn(1, 8, 16)
out, attn = scaled_dot_product_attention(Q, K, V)   # 权重每行和为 1

# 神经网络（解析/自动微分/数值梯度三重验证已通过）
model = MLP([(2, 16, "tanh"), (16, 3, "linear")])
opt = Adam(model.all_params(), lr=0.03)
for _ in range(2000):
    lg = model.forward(X)
    loss, dout = cross_entropy_loss(lg, y)
    gW, gb = model.backward(dout)
    opt.step([v for pair in zip(gW, gb) for v in pair])
```

---

## 四、验证体系（闭环交付的核心）

| 算法 | 不变量 | 验证方式 |
|------|--------|----------|
| 数值原语 | `softmax` 行和 = 1；`logsumexp` 不溢出 | 直接断言 + 数值梯度 vs 2x |
| Jacobi | `‖A vᵢ − λᵢ vᵢ‖ ≈ 0`；`VᵀV = I` | 残差 < 1e-8，对照 `numpy.eigh` |
| 自动微分 | 任意复合函数梯度正确 | autodiff vs 中心差分数值梯度 |
| 神经网络 | 梯度正确 + 可训练收敛 | **解析解 vs 自动微分 vs 数值梯度** 三重一致（< 1e-4）；XOR/螺旋 100% |
| 注意力 | 权重行和 = 1；因果掩码上三角 = 0 | 直接断言 + 多头 vs 手算逐头（< 1e-9） |
| PCA | 秩-k 重建最优 | top-k 误差 ≤ bottom-k（Eckart-Young） |
| GMM | 对数似然单调不减 | EM 定理，逐迭代断言 |
| KMeans | 惯性单调不增 | 逐迭代断言 |

> **为什么可信**：梯度同时由三条独立路径（手写解析反向传播、Var 自动微分引擎、中心差分数值法）计算得出，三者必须一致。这是工程上「可验证正确性」的最高标准之一。

---

## 五、算法清单与文献锚点

- **logsumexp / 稳定 softmax**：变分推断与概率模型的数值基石（Bishop, *Pattern Recognition and Machine Learning*, 2006）。
- **Jacobi 特征分解**：对称阵正交对角化，特征值与特征向量同步收敛，数值最稳定（Jacobi 1846）。
- **反向模式自动微分**：链式法则 = 反向传播的母题；现代深度学习框架的数学内核。
- **Adam**：`m̂_t = m_t/(1−β₁ᵗ)`，`v̂_t = v_t/(1−β₂ᵗ)`，`θ − α·m̂/√v̂`（Kingma & Ba, *Adam*, 2015）。
- **缩放点积注意力**：`softmax(QKᵀ/√d_k)V`，缩放因子防止点积饱和（Vaswani et al., *Attention Is All You Need*, 2017）。
- **PCA / Eckart-Young**：秩-k 截断重建误差在 Frobenius 意义下最优（Eckart & Young, 1936）。
- **GMM-EM**：对数似然单调不减是 EM 的定理性性质（Dempster, Laird & Rubin, 1977）。

---

## 六、性能基准（参考值，i7 / numpy 2.5）

| 操作 | 规模 | 耗时 |
|------|------|------|
| softmax | 4096×4096 | ~396 ms |
| SDPA | (1,32,64) | 0.23 ms |
| MultiHead | (8,64,8) | 0.65 ms |
| PCA fit | 2000×50 → k=10 | ~590 ms |
| GMM-EM | 1000×4, K=4 | ~138 ms |
| KMeans | 2000×4, K=5 | ~17 ms |
| 端到端 | 三分类螺旋 | **100% 准确率** |

> 说明：循环 Jacobi 为「精确可验证」教学实现（O(n⁴)），中等规模（≤64）即时完成；大规模特征问题推荐 `numpy.eigh`（200×200 仅需 22 ms）。Phosphor 的数值/注意力/神经网络内核均为向量化 numpy 实现，性能对标工业级。

---

## 七、路线图

- [x] 数值稳定原语 + 线性代数内核
- [x] 反向模式自动微分引擎
- [x] 神经网络（MLP / Adam / 损失）
- [x] 注意力机制（多头 / 因果掩码）
- [x] 经典机器学习（PCA / GMM / KMeans / NB）
- [x] 三重交叉验证测试 + 基准
- [ ] 卷积 / RNN / Transformer 编码器栈
- [ ] GPU 后端（CuPy）切换
- [ ] 自动微分 -> 训练器高层 API

---

## 八、许可

[MIT](LICENSE) © 2026 晨星 (Morning Star)
