# DDM

## 摘要

`DDM` 对伯努利预测误差流应用漂移检测方法。

## 更新 API

```python
result = rtta.DDM().update(error)
```

`update(...)` 每次接收一个 `error`；只推进状态时可调用 `advance(...)`。

## 工作原理

正输入视为分类误差。检测器以二项分布标准误为界，将当前误差过程与历史最佳基线比较，区分正常、警告和漂移状态。

## 递推公式

\[
p_t=\frac1t\sum_{i=1}^t\mathbf1[error_i>0],\qquad s_t=\sqrt{\frac{p_t(1-p_t)}t},\qquad m_t=p_t+s_t
\]

\[
m^*_t=\min_{i\le t}m_i,\qquad s^*_t=s_{\arg\min_i m_i}
\]

\[
y_t=\begin{cases}1,&m_t>m^*_t+d s^*_t\\0.5,&m_t>m^*_t+w s^*_t\\0,&\text{否则}\end{cases}
\]

发出漂移信号后，检测器会重置误差计数。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class DDM` 中实现。

## 参考资料

- [背景资料：概念漂移](https://en.wikipedia.org/wiki/Concept_drift)
