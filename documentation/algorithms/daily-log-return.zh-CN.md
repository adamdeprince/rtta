# DailyLogReturn

## 摘要

`DailyLogReturn` 计算相邻收盘价之间的对数收益率。

## 更新 API

```python
result = rtta.DailyLogReturn().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

对象保存上一收盘价，每次以当前值和上一值计算严格因果的单周期对数收益。

## 递推公式

\[
y_t=\log(close_t)-\log(close_{t-1})
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class DailyLogReturn` 中实现。

## 参考资料

- [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
