# CumulativeReturn

## 摘要

`CumulativeReturn` 计算相对第一个收盘价的累计收益率。

## 更新 API

```python
result = rtta.CumulativeReturn().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

第一次更新保存基准收盘价；此后每次调用只接收一个新观测，并返回当前收盘价相对该基准的收益率。

## 递推公式

令 \(z_t=close_t\) 为一次更新接收的观测。

\[
y_t=\frac{close_t}{close_0}-1
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CumulativeReturn` 中实现。

## 参考资料

- [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
