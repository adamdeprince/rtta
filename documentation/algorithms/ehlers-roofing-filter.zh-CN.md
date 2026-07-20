# Ehlers Roofing 滤波器（EhlersRoofingFilter）

## 摘要

`EhlersRoofingFilter` 是 RTTA 对 Roofing 滤波器的流式实现：先由二极点高通级移除长期漂移，再以 Ehlers Super Smoother 低通级保留中频周期内容。所得带限序列可作为经过降噪的振荡器输入。

## 更新 API

```python
result = rtta.EhlersRoofingFilter(hp_period=48, lp_period=10, fillna=True).update(price)
```

| 参数 | 默认值 | 含义 |
|-------------|---------|---------|
| `hp_period` | `48`    | 高通临界周期（最小为 2） |
| `lp_period` | `10`    | Super Smoother 低通周期（最小为 2） |
| `fillna`    | `True`  | 若为 `False`，取得 `hp_period` 个样本前返回 NaN |

`update(...)` 返回的结果包含：

- `roof`——带通（Roofing）输出
- `highpass`——中间高通序列

`advance(price)` 更新状态但不返回新的 Python 对象；`last()` 返回最近一次结果。

## 工作原理

价格序列混合了缓慢趋势（极低频）、中间周期和高频噪声。Roofing 滤波器按两步处理：

1. 以接近 `hp_period` 的周期作**高通滤波**，移除趋势，使残差围绕零轴振荡。
2. 以 `lp_period` 周期的 **Super Smoother 低通滤波**，移除目标周期频带以上的残余噪声。

高通级采用 Ehlers 二极点形式，系数 \(\alpha\) 由角度 \(0.707\cdot2\pi/P_{hp}\) 导出（\(0.707\approx1/\sqrt2\) 因子用于设定二极点高通响应）。低通级复用与 [`EhlersSuperSmoother`](ehlers-super-smoother.zh-CN.md) 相同的系数构造，但处理高通序列而不是原始价格。

## 递推公式

### 高通系数

令 \(P_h=\max(\texttt{hp\_period},2)\)：

\[
\theta_h = \frac{0.707 \cdot 2\pi}{P_h},\qquad
\alpha = \frac{\cos\theta_h + \sin\theta_h - 1}{\cos\theta_h}
\]

### Super Smoother 系数（低通）

令 \(P_\ell=\max(\texttt{lp\_period},2)\)：

\[
\theta_\ell = \frac{\sqrt{2}\,\pi}{P_\ell},\qquad
a_1 = e^{-\theta_\ell},\qquad
b_1 = 2 a_1 \cos(\theta_\ell)
\]

\[
c_2 = b_1,\qquad
c_3 = -a_1^{2},\qquad
c_1 = 1 - c_2 - c_3
\]

### 高通级

前两个样本取 \(HP_t=0\)。此后令 \(k=1-\alpha/2\)：

\[
\begin{aligned}
HP_t &= k^{2}\,(x_t - 2 x_{t-1} + x_{t-2}) \\
&\quad + 2(1-\alpha)\, HP_{t-1} \\
&\quad - (1-\alpha)^{2}\, HP_{t-2}
\end{aligned}
\]

### Roofing 级（对高通结果应用 Super Smoother）

前两个样本取 \(R_t=HP_t\)。此后：

\[
R_t = c_1 \cdot \frac{HP_t + HP_{t-1}}{2} + c_2\, R_{t-1} + c_3\, R_{t-2}
\]

结果字段：`roof` \(=R_t\)，`highpass` \(=HP_t\)。

当 `fillna=False` 且已取得的样本少于 \(P_h\) 个时，两个字段均为 NaN。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class EhlersRoofingFilter` 中实现。
- 结果类型：`EhlersRoofingFilterResult`（`roof`、`highpass`）。
- 高通值和 Roofing 值各保留两个滞后状态（`hp1_`/`hp2_`、`roof1_`/`roof2_`），另保留两个价格滞后状态以计算二阶差分。
- 批量辅助函数：`batch_ehlers_roofing`。

## 参考资料

- [MESA Software——John Ehlers 论文](https://www.mesasoftware.com/)
- [Ehlers Roofing / 带通滤波器讨论](https://www.mesasoftware.com/papers/GaussianFilters.pdf)
