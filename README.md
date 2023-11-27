RTTA
======================

Purpose
-------

The purpose of this package is to implement a very low latency
incremental technical analysis toolkit.  Most technical analysis
tool-kits work in a "batch mode" where you hand them a blob of data and
in a pandas series and they return a series with the computed data.
Incremental updates for these require O(n) work.  There is one tool,
[talipp](https://pypi.org/project/talipp/) that is designed to support
incremental updates, but it is implemented in pure python and is a
little more than an order of magnitude slower than rtta.  On a 5995WX
talipp's exponential moving average requires 465ns; rtta's requires
36ns.  A bare python function call requires 35ns, so we're about as
fast as fast can be.

Installation
------------


```bash
pip install rtta
```

Usage
-----

Each operator has a paramater fillna.  If set to false, nan values
will be returned until the operation is "populated".  If set to true,
best guesses will be returned until the operation is populated.

So for example, our simple moving average `SMAIndicator` works sort of like this:

```python
>>> import rtta.trend as trend
>>> sma = trend.SMAIndicator(window=4, fillna=True)
>>> sma.update(1)
1
>>> sma.update(2)
1.5
>>> sma.update(3)
2
>>> sma.update(2)
2
>>> sma.update(2)
2.25 <- The 1 fell off the end of the sliding window
```

Performance
-----------

|Indicator | Latency |
|----------|---------|
| SMA      | 36ns    |
| EMA      | 36ns    |
| MACD     | 55ns    |