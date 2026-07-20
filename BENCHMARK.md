# Benchmarks

The benchmark pages report the live per-tick Python-facing path:

- `advance(...)`: update state only.
- `update(...)`: update state and return a Python value/result.

These runs were collected on 2026-07-19 over SSH. Hostnames are intentionally omitted from the documentation; each full result page is identified by CPU type.

Each CPU page carries its own system metadata and full 247-algorithm latency table.

Run command:

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## Results By CPU

- [Apple M4 Max](documentation/benchmarks/apple-m4-max.md): `macOS-26.5.1-arm64-arm-64bit-Mach-O`, `arm64`, Python `3.14.5`, NumPy `2.5.1`, RTTA `0.2.3`; median `advance(...)` **29.9 ns/update**, median `update(...)` **41.1 ns/update**.
- [Intel Xeon 6975P-C](documentation/benchmarks/intel-xeon-6975p-c.md): `Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`, `x86_64`, Python `3.14.4`, NumPy `2.5.1`, RTTA `0.2.3`; median `advance(...)` **39.2 ns/update**, median `update(...)` **52.2 ns/update**.
- [Loongson-3A6000](documentation/benchmarks/loongson-3a6000.md): `Linux-5.4.18-167-generic-loongarch64-with-glibc2.28`, `loongarch64`, Python `3.14.4`, NumPy `2.5.1`, RTTA `0.2.3`; median `advance(...)` **104 ns/update**, median `update(...)` **139 ns/update**.
