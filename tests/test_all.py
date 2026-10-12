"""Phosphor 全量不变量测试 —— 闭环验证核心。

验证哲学（世界顶级严谨度）：
- 每条算法至少一条可被独立实现交叉验证的「不变量」。
- 神经网络梯度：解析解 vs 自动微分 vs 数值梯度 三重交叉验证。
- 任一失败即非零退出，保证「可复现、可验证」。

运行：
    python tests/test_all.py
作者：晨星
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import phosphor as ph
from phosphor import (
    logsumexp, softmax, numerical_gradient, jacobi_eigen, power_iteration, top_eigen,
    cholesky, Var, MLP, AutodiffMLP, Adam, cross_entropy_loss, scaled_dot_product_attention,
    MultiHeadAttention, PCA, GMM, KMeans, GaussianNB,
)

PASS, FAIL = 0, 0
FAILS = []


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        FAILS.append(name)
        print(f"  ❌ {name}  {detail}")


def rel_err(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


print("\n=== 1. 数值稳定原语 ===")
# logsumexp 防溢出：超大值不报 nan/inf
big = np.array([1000.0, 1000.0, 1000.0])
ls = logsumexp(big)
check("logsumexp 不溢出", np.isfinite(ls) and abs(ls - (1000 + np.log(3))) < 1e-6,
      f"ls={ls}")
# softmax 行和为 1
x = np.random.randn(5, 7)
sm = softmax(x)
check("softmax 行和为1", np.allclose(sm.sum(axis=1), 1.0, atol=1e-12))
# 数值梯度 vs 解析导数 f(x)=Σx² → 2x
f = lambda v: np.sum(v ** 2)
v = np.array([1.0, 2.0, -3.0, 0.5])
ng = numerical_gradient(f, v)
check("数值梯度 vs 解析导数 (2x)", rel_err(ng, 2 * v) < 1e-5, f"ng={ng}")

print("\n=== 2. 线性代数 ===")
rng = np.random.default_rng(0)


def rn(*args):
    """rn 的 *args 包装（支持多维度写法）。"""
    shape = args[0] if (len(args) == 1 and isinstance(args[0], (tuple, list))) else tuple(args)
    return rng.standard_normal(shape)
A = rn(6, 6)
S = A + A.T  # 对称
evals, evecs = jacobi_eigen(S)
# 特征方程残差
res = max(np.linalg.norm(S @ evecs[:, i] - evals[i] * evecs[:, i]) for i in range(6))
check("Jacobi 特征方程残差 ||Av-λv||≈0", res < 1e-8, f"res={res:.2e}")
# 正交归一
ortho = np.max(np.abs(evecs.T @ evecs - np.eye(6)))
check("Jacobi 特征向量正交归一 VᵀV=I", ortho < 1e-8, f"ortho={ortho:.2e}")
# 与 numpy 对照
np_evals = np.sort(np.linalg.eigvalsh(S))[::-1]
check("Jacobi 特征值 vs numpy", rel_err(np.sort(evals)[::-1], np_evals) < 1e-8)
# 幂迭代主特征（返回模最大特征值）
lam, vec = power_iteration(S)
check("幂迭代主特征残差", np.linalg.norm(S @ vec - lam * vec) < 1e-8)
mag_max = np.sort(np.abs(evals))[::-1][0]
check("幂迭代主特征值(模最大) vs Jacobi", abs(abs(lam) - mag_max) < 1e-6)
# top_eigen：返回模最大的前 3 个特征对
tvals, tvecs = top_eigen(S, 3)
order = np.argsort(np.abs(evals))[::-1][:3]
target = np.sort(np.abs(evals[order]))[::-1]
got = np.sort(np.abs(tvals))[::-1]
check("top_eigen 对应 Jacobi 模最大特征", rel_err(got, target) < 1e-5,
      f"got={got} target={target}")
check("top_eigen 向量正交归一",
      np.max(np.abs(tvecs.T @ tvecs - np.eye(3))) < 1e-6)
# Cholesky 重建
P = rn(5, 5)
SPD = P @ P.T + 5 * np.eye(5)
L = cholesky(SPD)
check("Cholesky L Lᵀ≈A", np.max(np.abs(L @ L.T - SPD)) < 1e-10)

print("\n=== 3. 自动微分 ===")
# f = Σ((Wx+b)²) 梯度 vs 数值梯度
W = rn(3, 2)
b = rn(3)
xx = rn(2)
def ff(v):
    Wp = v[:6].reshape(3, 2); bp = v[6:9]
    return np.sum((Wp @ xx + bp) ** 2)
vec = np.concatenate([W.ravel(), b])
ng2 = numerical_gradient(ff, vec)
# 自动微分
xv = Var(xx)
Wv, bv = Var(W), Var(b)
y = (Wv.matmul(xv) + bv)
loss = (y * y).sum()
loss.backward()
ag = np.concatenate([Wv.grad.ravel(), bv.grad.ravel()])
check("autodiff vs 数值梯度", rel_err(ag, ng2) < 1e-5, f"re={rel_err(ag,ng2):.2e}")
# softmax + 交叉熵梯度（autodiff）
logits_v = Var(rn(4))
p = logits_v.softmax()
# 手工构造 onehot 选择: 用 sum(p*onehot)
oh = Var(np.eye(4)[2])
ce = -(oh * p.log()).sum()
ce.backward()
# 解析 CE 梯度: p - onehot
analytic_ce = p.data.copy(); analytic_ce[2] -= 1
check("softmax 交叉熵梯度", rel_err(logits_v.grad, analytic_ce) < 1e-9)

print("\n=== 4. 神经网络（三重交叉验证）===")
layers = [(2, 5, "tanh"), (5, 3, "linear")]
mlp = MLP(layers)
Xs = rn(4, 2)
ys = np.array([0, 1, 2, 1])
# (a) 解析梯度
logits = mlp.forward(Xs)
loss, dout = cross_entropy_loss(logits, ys)
gW, gb = mlp.backward(dout)
ag_flat = np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in zip(gW, gb)])
# (b) 自动微分梯度（同源参数）
adm = AutodiffMLP(layers, mlp)
ce_var = adm.cross_entropy(Xs, ys)
ce_var.backward()
ad_flat = np.concatenate([np.concatenate([adm.W[i].grad.ravel(), adm.b[i].grad.ravel()])
                          for i in range(len(layers))])
check("MLP 解析梯度 vs 自动微分", rel_err(ag_flat, ad_flat) < 1e-6,
      f"re={rel_err(ag_flat, ad_flat):.2e}")
# (c) 数值梯度
def loss_fn(v):
    mlp.unflatten(v)
    lg, _ = cross_entropy_loss(mlp.forward(Xs), ys)
    return lg
num_flat = numerical_gradient(loss_fn, mlp.flat_params())
check("MLP 解析梯度 vs 数值梯度", rel_err(ag_flat, num_flat) < 1e-4,
      f"re={rel_err(ag_flat, num_flat):.2e}")

# XOR 训练收敛
xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
xor_y = np.array([0, 1, 1, 0])
xnet = MLP([(2, 8, "tanh"), (8, 2, "linear")])
opt = Adam(xnet.all_params(), lr=0.05)
losses = []
for _ in range(2000):
    lg = xnet.forward(xor_x)
    loss, dout = cross_entropy_loss(lg, xor_y)
    gW, gb = xnet.backward(dout)
    grads = []
    for w, b in zip(gW, gb):
        grads.append(w)
        grads.append(b)
    opt.step(grads)
    losses.append(loss)
check("XOR 训练损失单调下降", losses[-1] < losses[0] * 0.1,
      f"init={losses[0]:.3f} final={losses[-1]:.4f}")
pred = np.argmax(xnet.forward(xor_x), axis=1)
check("XOR 分类全对", np.array_equal(pred, xor_y), f"pred={pred}")

print("\n=== 5. 注意力机制 ===")
T = 5
Q = rn(1, T, 4)
K = rn(1, T, 4)
V = rn(1, T, 4)
out, attn = scaled_dot_product_attention(Q, K, V)
check("注意力权重行和为1", np.allclose(attn.sum(axis=-1), 1.0, atol=1e-12))
# 因果掩码
out_c, attn_c = scaled_dot_product_attention(Q, K, V, causal=True)
upper = np.triu(attn_c[0], k=1)  # 提取严格上三角
check("因果掩码上三角权重=0", np.all(upper == 0), f"max_upper={upper.max():.2e}")
# 多头 vs 手算逐头
mha = MultiHeadAttention(d_model=8, n_heads=2)
# 构造匹配输入
Xm = rn(1, T, 8)
mha_out2 = mha.forward(Xm)
# 手算对照
Wq, Wk, Wv, Wo = mha.params()
Qh = (Xm @ Wq).reshape(1, T, 2, 4).transpose(0, 2, 1, 3)
Kh = (Xm @ Wk).reshape(1, T, 2, 4).transpose(0, 2, 1, 3)
Vh = (Xm @ Wv).reshape(1, T, 2, 4).transpose(0, 2, 1, 3)
heads = []
for i in range(2):
    oi, _ = scaled_dot_product_attention(Qh[:, i], Kh[:, i], Vh[:, i])
    heads.append(oi)
manual = np.stack(heads, 1).transpose(0, 2, 1, 3).reshape(1, T, 8) @ Wo
check("多头注意力 vs 手算", np.max(np.abs(mha_out2 - manual)) < 1e-9)

print("\n=== 6. 经典机器学习 ===")
# PCA Eckart-Young：top-k 重建误差 ≤ bottom-k
data = rn(200, 10) @ rn(10, 10) * np.linspace(1, 0.1, 10)
pca_top = PCA(n_components=3).fit(data)
err_top = pca_top.reconstruction_error(data)
# bottom-k（取最小特征方向）
pca_bot = PCA(3)
pca_bot.fit(data)
# 反转成分取最小 3 个
evals, evecs = jacobi_eigen(np.cov(data.T))
bot_comps = evecs[:, -3:].T
pca_bot.components_ = bot_comps
err_bot = pca_bot.reconstruction_error(data)
check("PCA Eckart-Young (top-k ≤ bottom-k)", err_top <= err_bot + 1e-6,
      f"top={err_top:.3f} bot={err_bot:.3f}")

# GMM：对数似然单调 + 责任度行和=1 + 可分数据预测正确
g1 = rn(100, 2) + 5
g2 = rn(100, 2) - 5
gmm_X = np.vstack([g1, g2])
gmm_y = np.array([0] * 100 + [1] * 100)
gmm = GMM(n_components=2, seed=1).fit(gmm_X)
ll_mono = all(gmm.ll_history_[i + 1] >= gmm.ll_history_[i] - 1e-6
              for i in range(len(gmm.ll_history_) - 1))
check("GMM 对数似然单调不减", ll_mono, f"ll={gmm.ll_history_[-3:]}")
resp = gmm.responsibilities(gmm_X)
check("GMM 责任度行和=1", np.allclose(resp.sum(axis=1), 1.0, atol=1e-9))
gmm_acc = np.mean((gmm.predict(gmm_X) == gmm_y))
check("GMM 可分数据聚类正确", gmm_acc > 0.95, f"acc={gmm_acc:.2f}")

# KMeans：惯性单调不增
km = KMeans(n_clusters=2, seed=2).fit(gmm_X)
inertia_mono = all(km.inertia_history_[i + 1] <= km.inertia_history_[i] + 1e-6
                   for i in range(len(km.inertia_history_) - 1))
check("KMeans 惯性单调不增", inertia_mono, f"inertia={km.inertia_history_[-3:]}")
km_acc = np.mean((km.labels_ == gmm_y))
check("KMeans 可分数据聚类正确", km_acc > 0.95 or km_acc < 0.05, f"acc={km_acc:.2f}")

# 高斯朴素贝叶斯
nb_X = np.vstack([rn(80, 3) + 3, rn(80, 3) - 3])
nb_y = np.array([0] * 80 + [1] * 80)
nb = GaussianNB().fit(nb_X, nb_y)
nb_acc = np.mean(nb.predict(nb_X) == nb_y)
check("高斯朴素贝叶斯可分数据全对", nb_acc == 1.0, f"acc={nb_acc:.2f}")

print(f"\n{'='*40}")
print(f"结果：{PASS} 通过 / {FAIL} 失败")
if FAIL:
    print("失败项：", FAILS)
    sys.exit(1)
print("🎉 全部不变量验证通过")
