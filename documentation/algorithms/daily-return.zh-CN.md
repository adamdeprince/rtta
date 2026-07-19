# DailyReturn

## 摘要

`DailyReturn` 计算相邻收盘价之间的百分比收益率。

## 更新 API

```python
result = rtta.DailyReturn().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

对象保存所需的历史收盘价，并按因果顺序计算当前相对过去值的收益率。

## 递推公式

\[
y_t=\frac{close_t-close_{t-n}}{close_{t-n}}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class DailyReturn` 中实现。

## 参考资料

- [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
