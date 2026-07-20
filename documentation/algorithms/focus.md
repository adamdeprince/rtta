# FOCuS

## Summary

`FOCuS` is a two-sided **Functional Online CUSUM** mean changepoint detector with
candidate pruning in the style of Romano, Eckley, Fearnhead, and Rigaill. Each update
returns a signal in \(\{-1,0,+1\}\) and the maximum likelihood-ratio statistic over
active candidates.

## Update API

```python
import rtta

ind = rtta.FOCuS(threshold=10.0, mu0=0.0, sigma=1.0, max_candidates=200)
result = ind.update(value)
# result.signal ∈ {-1, 0, +1}, result.statistic ≥ 0
```

`advance(...)` updates state without returning a result. After a fire (`signal ≠ 0`),
the candidate set is cleared so detection restarts.

## Theory Of Operation

FOCuS maintains a pruned set of candidate changepoint locations. For Gaussian
observations with known pre-change mean \(\mu_0\) and variance \(\sigma^2\), each
candidate stores the cumulative centered sum and length since the putative change. The
test statistic for a candidate with sum \(S\) and length \(n\) is the Gaussian mean
GLR:

\[
\Lambda = \frac{S^2}{2\sigma^2 n}.
\]

Functional pruning removes dominated candidates (by mean order and shorter length),
keeping cost linear in a small number of candidates (`max_candidates` cap per side).
When \(\max \Lambda \ge h\), the detector emits the sign of the winning sum and resets.

## Recurrence

Center the observation: \(y_t = x_t - \mu_0\). Let the candidate set at \(t-1\) be
pairs \((S^{(j)}, n^{(j)})\). Form the updated multiset

\[
\mathcal{C}'_t = \bigl\{(y_t, 1)\bigr\}
\cup
\bigl\{(S^{(j)} + y_t,\ n^{(j)}+1)\bigr\}_j.
\]

Prune \(\mathcal{C}'_t\) separately on positive and negative sums: sort by mean
\(S/n\) (ascending for positive, descending for negative) and keep only candidates
with strictly decreasing length (dominance prune), then cap each side at
`max_candidates`. Denote the pruned set \(\mathcal{C}_t\).

Statistic and signal:

\[
\Lambda_t = \max_{(S,n)\in\mathcal{C}_t,\,n>0}
\frac{S^2}{2\sigma^2 n}
\quad
\bigl(\text{implemented as } S^2 \cdot (1/(2\sigma^2)) / n\bigr),
\]

\[
\mathrm{signal}_t =
\begin{cases}
\operatorname{sign}(S^\star), & \Lambda_t \ge h,\\
0, & \text{otherwise},
\end{cases}
\]

where \(S^\star\) is the sum of the maximizing candidate (\(S^\star \ge 0 \Rightarrow +1\)).
On a fire, \(\mathcal{C}_t \leftarrow \emptyset\). Variance is floored:
\(\sigma^2 \leftarrow \max(\sigma^2, 10^{-18})\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class FOCuS`
(`prune_candidates`). `ResidualFOCuS` is a thin wrapper that feeds residuals into the
same engine. Canonical doc path is `focus.md` (not `fo-cu-s.md`).

## Reference

- [Romano, Eckley, Fearnhead & Rigaill, “Fast Online Changepoint Detection via Functional Pruning CUSUM Statistics” (JMLR / arXiv)](https://arxiv.org/abs/2110.08205)
