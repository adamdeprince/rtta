# NadarayaWatsonEnvelope

## 摘要

`NadarayaWatsonEnvelope` 是带加权残差带的高斯核 Nadaraya-Watson 平滑器。

## 更新 API

```python
result = rtta.NadarayaWatsonEnvelope(window=32).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标在滚动窗口内按时间距离施加高斯核权重，以因果方式计算平滑值，并由加权残差构造包络。

## 递推公式

\[
w_{t,i}=\exp\left(-\frac{(t-i)^2}{2h^2}\right)
\]

\[
\hat x_t=\frac{\sum_{i\in W_t}w_{t,i}x_i}{\sum_{i\in W_t}w_{t,i}}
\]

`update(...)` 返回含 `middle`、`upper` 和 `lower` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class NadarayaWatsonEnvelope` 中实现。

## 参考资料

- [Nadaraya-Watson 核回归](https://classic.d2l.ai/chapter_attention-mechanisms/nadaraya-watson.html)
