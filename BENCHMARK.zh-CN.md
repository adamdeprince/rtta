# 基准测试

基准测试页面报告面向 Python 的实时逐 tick 路径：

- `advance(...)`：仅更新状态。
- `update(...)`：更新状态并返回 Python 值/结果。

这些结果采集于 2026-07-19（含 SSH 远端）。公开文档有意省略主机名；每个完整结果页以 CPU 类型标识。

每个 CPU 页面包含各自的系统元数据以及完整的 247 项算法延迟表。

运行命令：

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## 按 CPU 的结果

- [Apple M4 Max](documentation/benchmarks/apple-m4-max.zh-CN.md)：`macOS-26.5.1-arm64-arm-64bit-Mach-O`，`arm64`，Python `3.14.5`，NumPy `2.5.1`，RTTA `0.2.3`；`advance(...)` 中位数为 **29.9 ns/update**，`update(...)` 中位数为 **41.1 ns/update**。
- [Intel Xeon 6975P-C](documentation/benchmarks/intel-xeon-6975p-c.zh-CN.md)：`Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`，`x86_64`，Python `3.14.4`，NumPy `2.5.1`，RTTA `0.2.3`；`advance(...)` 中位数为 **39.2 ns/update**，`update(...)` 中位数为 **52.2 ns/update**。
- [Loongson-3A6000](documentation/benchmarks/loongson-3a6000.zh-CN.md)：`Linux-5.4.18-167-generic-loongarch64-with-glibc2.28`，`loongarch64`，Python `3.14.4`，NumPy `2.5.1`，RTTA `0.2.3`；`advance(...)` 中位数为 **104 ns/update**，`update(...)` 中位数为 **139 ns/update**。
