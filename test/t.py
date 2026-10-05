import numpy as np
import matplotlib.pyplot as plt
from math import erf

# ============ 基础工具 ============
SQRT_2 = np.sqrt(2.0)
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_erf = np.vectorize(erf)          # 向量化 math.erf，避免依赖 scipy


# ============ 激活函数定义 ============
def gelu_exact(x):
    """精确 GELU: x * Φ(x)，Φ 为标准正态分布 CDF"""
    return x * 0.5 * (1.0 + _erf(x / SQRT_2))


def gelu_approx(x):
    """tanh 近似版 GELU（BERT 原论文使用的形式）"""
    return 0.5 * x * (1.0 + np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)))


def gelu_exact_grad(x):
    """精确 GELU 的导数: Φ(x) + x·φ(x)"""
    cdf = 0.5 * (1.0 + _erf(x / SQRT_2))
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


def gelu_approx_grad(x):
    """tanh 近似版的导数（解析式）"""
    u = SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def silu(x):
    """Swish-1: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-x))


def silu_grad(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1 + x * (1 - s))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


# ============ 绘图 ============
x = np.linspace(-5, 5, 1000)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("GELU vs Other Activation Functions", fontsize=15, fontweight="bold")

# ---- (1) 激活函数对比 ----
ax = axes[0, 0]
ax.plot(x, gelu_exact(x), lw=2.5, color="crimson", label="GELU (exact)")
ax.plot(x, relu(x), lw=2.0, ls="--", color="steelblue", label="ReLU")
ax.plot(x, silu(x), lw=2.0, ls="-.", color="seagreen", label="SiLU / Swish")
ax.plot(x, sigmoid(x), lw=1.5, ls=":", color="gray", label="Sigmoid")
ax.plot(x, np.tanh(x), lw=1.5, ls=":", color="orange", label="Tanh")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
ax.set_title("Activation Functions f(x)")
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ---- (2) GELU 精确版 vs 近似版 ----
ax = axes[0, 1]
ax.plot(x, gelu_exact(x), lw=3.5, color="lightcoral", label="GELU exact")
ax.plot(x, gelu_approx(x), lw=1.5, ls="--", color="navy", label="GELU tanh-approx")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
ax.set_title("GELU: exact vs tanh approximation")
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 标注负值区间的最小值（约 -0.17 @ x ≈ -0.75）
x_min = x[np.argmin(gelu_exact(x))]
y_min = gelu_exact(x_min)
ax.annotate(f"min ≈ {y_min:.3f}\n@ x ≈ {x_min:.2f}",
            xy=(x_min, y_min), xytext=(x_min + 1.0, y_min - 0.6),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

# ---- (3) 导数对比 ----
ax = axes[1, 0]
ax.plot(x, gelu_exact_grad(x), lw=2.5, color="crimson", label="GELU'")
ax.plot(x, relu_grad(x), lw=2.0, ls="--", color="steelblue", label="ReLU'")
ax.plot(x, silu_grad(x), lw=2.0, ls="-.", color="seagreen", label="SiLU'")
ax.plot(x, sigmoid_grad(x), lw=1.5, ls=":", color="gray", label="Sigmoid'")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_title("Derivatives f'(x)")
ax.set_xlabel("x"); ax.set_ylabel("f'(x)")
ax.set_ylim(-0.3, 1.4)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ---- (4) 近似误差 ----
ax = axes[1, 1]
err = gelu_approx(x) - gelu_exact(x)
ax.plot(x, err, lw=2, color="purple")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_title(f"Approximation Error (max |err| = {np.abs(err).max():.2e})")
ax.set_xlabel("x"); ax.set_ylabel("approx - exact")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gelu_visualization.png", dpi=150, bbox_inches="tight")
plt.show()