# Ehlers 去周期器（EhlersDecycler）

## 摘要

`EhlersDecycler` 是 RTTA 对 Ehlers 去周期器的流式实现。它估计价格中的低频**去周期分量**（类似趋势的平滑线），并输出强调被移除高频内容的**振荡器**残差 \(x_t-\text{decycle}_t\)。

## 更新 API

```python
result = rtta.EhlersDecycler(hp_period=60, fillna=True).update(price)
```

| 参数 | 默认值 | 含义 |
|-------------|---------|---------|
| `hp_period` | `60`    | 控制高通 / 去周期截止频率的周期（最小为 2） |
| `fillna`    | `True`  | 若为 `False`，取得 `hp_period` 个样本前返回 NaN |

`update(...)` 返回：

- `decycle`——低频去周期估计值
- `oscillator`——\(price-decycle\)

`advance(price)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

Ehlers 去周期器通过从价格中减去单极点高通分量得到（等价于互补低通滤波器）。高通系数 \(\alpha\) 由周期 \(P\) 导出，低频路径为：

\[
D_t = \frac{\alpha}{2}(x_t + x_{t-1}) + (1-\alpha) D_{t-1}
\]

因此，\(D_t\) 跟踪缓慢变化的结构，而残差 \(x_t-D_t\) 的表现类似于快速波动的零均值振荡器。增大 `hp_period` 会把更多频谱纳入去周期分量，使振荡器更平静；减小周期则使去周期线更紧密地贴合价格。

它比二极点 Roofing 滤波器更简单：只需要一个 \(\alpha\)，以及各一阶的价格与去周期状态。

## 递推公式

令 \(P=\max(\texttt{hp\_period},2)\)：

\[
\theta = \frac{2\pi}{P},\qquad
\alpha = \frac{\cos\theta + \sin\theta - 1}{\cos\theta}
\]

首个样本（\(t=0\)）：

\[
D_0 = x_0
\]

此后：

\[
D_t = \frac{\alpha}{2}\,(x_t + x_{t-1}) + (1-\alpha)\, D_{t-1}
\]

\[
O_t = x_t - D_t
\]

结果：`decycle` \(=D_t\)，`oscillator` \(=O_t\)。

当 `fillna=False` 且已处理的样本少于 \(P\) 个时，两个字段均为 NaN。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class EhlersDecycler` 中实现。
- 结果类型：`EhlersDecyclerResult`（`decycle`、`oscillator`）。
- 状态：`price1_`、`decycle1_`、样本 `count_`。
- 注意：\(\theta=2\pi/P\)（没有 \(0.707\) 因子），不同于 [`EhlersRoofingFilter`](ehlers-roofing-filter.zh-CN.md) / Cyber Cycle 高通滤波器。
- 批量辅助函数：`batch_ehlers_decycler`。

## 参考资料

- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
- [去周期器 / 高通互补滤波器](https://www.mesasoftware.com/papers/GaussianFilters.pdf)
