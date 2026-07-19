# FeatureDistributionDriftDetector

## 摘要

`FeatureDistributionDriftDetector` 是针对单一流式特征分布、内存有界的 ADWIN 式漂移检测器。

## 更新 API

```python
result = rtta.FeatureDistributionDriftDetector().update(feature)
```

`update(...)` 每次接收一个 `feature`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器维护自适应近期窗口，检查每个允许的切分点，并判断新旧子窗口的均值差异是否显著。接受切分后丢弃较早的前缀，信号符号表示均值变化方向。

## 递推公式

\[
W_t=\operatorname{tail}_{max\_window}(W_{t-1}\cup\{x_t\})
\]

\[
\epsilon(c)=R_t\sqrt{\frac12\log\left(\frac4\delta\right)\left(\frac1c+\frac1{|W_t|-c}\right)}
\]

\[
c^*=\arg\max_c|\bar x_{c:|W_t|}-\bar x_{1:c}|\quad\text{s.t.}\quad|\bar x_{c:|W_t|}-\bar x_{1:c}|>\epsilon(c)
\]

\[
y_t=\begin{cases}\operatorname{sgn}(\bar x_{c^*:|W_t|}-\bar x_{1:c^*}),&c^*\text{ 存在}\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class FeatureDistributionDriftDetector` 中实现。

## 参考资料

- [背景资料：概念漂移](https://en.wikipedia.org/wiki/Concept_drift)
