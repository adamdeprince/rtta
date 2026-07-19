# 基准测试

基准测试页面测量面向 Python 的实时逐笔处理路径：

- `advance(...)`：只更新状态。
- `update(...)`：更新状态，并返回 Python 值或结果对象。

这些测试于 2026 年 6 月 9 日通过 SSH 运行。文档有意省略主机名；每份完整结果均以 CPU 型号标识。

每个 CPU 页面都包含对应的系统元数据，以及全部 188 种算法的延迟表。

运行命令：

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## 各 CPU 的测试结果

- [Intel Xeon 6975P-C](documentation/benchmarks/intel-xeon-6975p-c.zh-CN.md)：`Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`、`x86_64`、Python `3.14.4`、NumPy `2.4.6`、RTTA `0.2.1`；`advance(...)` 延迟中位数为 **51.6 ns/update**，`update(...)` 延迟中位数为 **59.4 ns/update**。
- [Apple M4 Max](documentation/benchmarks/apple-m4-max.zh-CN.md)：`macOS-15.3.1-arm64-arm-64bit-Mach-O`、`arm64`、Python `3.14.5`、NumPy `2.4.6`、RTTA `0.2.1`；`advance(...)` 延迟中位数为 **37.9 ns/update**，`update(...)` 延迟中位数为 **43.2 ns/update**。
- [Loongson-3A6000](documentation/benchmarks/loongson-3a6000.zh-CN.md)：`Linux-5.4.18-110-generic-loongarch64-with-glibc2.28`、`loongarch64`、Python `3.14.4`、NumPy `2.4.6`、RTTA `0.2.1`；`advance(...)` 延迟中位数为 **121 ns/update**，`update(...)` 延迟中位数为 **152 ns/update**。
