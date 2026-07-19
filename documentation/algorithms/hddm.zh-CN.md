# HDDM

## 摘要

`HDDM` 使用 Hoeffding 界检测伯努利预测误差流的漂移。

## 更新 API

```python
result = rtta.HDDM().update(error)
```

`update(...)` 每次接收一个 `error`；只推进状态时可调用 `advance(...)`。

## 工作原理

正输入视为分类误差。检测器根据 Hoeffding 界把当前误差过程与历史最佳基线比较，区分正常、警告和漂移状态。

## 递推公式

\[
\bar e_t=\frac1t\sum_{i=1}^t\mathbf1[error_i>0],\qquad b_t(\delta)=\sqrt{\frac{\log(1/\delta)}{2t}}
\]

\[
(\bar e^*_t,b^*_t)=\arg\min_{i\le t}(\bar e_i+b_i(\delta_{drift}))
\]

\[
y_t=\begin{cases}1,&\bar e_t-\bar e^*_t>b_t(\delta_{drift})+b^*_t\\0.5,&\bar e_t-\bar e^*_t>b_t(\delta_{warning})+b^*_t\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class HDDM` 中实现。

## 参考资料

- [背景资料：Hoeffding 不等式](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)
