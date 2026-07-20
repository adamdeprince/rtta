"""FourierResidueIdentity: streaming Fourier-Residue Identity channels.

Reference: V. Portnaya, "The Bounce Has No Direction: Sign, Magnitude, and the
Microstructure of Equity Return Predictability", arXiv:2606.29591 (June 2026).
"""

import math

import numpy as np
import pytest


FIELDS = (
    "rho",
    "rho_sign",
    "rho_magnitude",
    "z_rho",
    "z_sign",
    "directional_share",
    "elliptical_ratio",
    "variance_ratio",
    "variance_ratio_sign",
    "variance_ratio_magnitude",
    "z_variance_ratio",
    "persistence",
    "signal",
    "score",
    "magnitude_forecast",
)

# Effectively-expanding memory, so the streaming estimator reproduces the
# full-sample statistics the paper reports.
FULL_SAMPLE = dict(span=1e9, median_window=8192, max_lag=8)


def _prices(returns):
    return 100.0 * np.exp(np.cumsum(np.asarray(returns, dtype=float)))


def _ar1(n, phi, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def _roll_bounce(n, half_spread=0.003, sigma=0.01, seed=0):
    """Roll (1984): martingale efficient price plus an IID bid-ask bounce."""
    rng = np.random.default_rng(seed)
    efficient = rng.normal(0.0, sigma, n)
    side = rng.choice([-1.0, 1.0], n)
    prev_side = np.concatenate([[side[0]], side[:-1]])
    return efficient + half_spread * (side - prev_side)


def _reference_rho(returns, lag):
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    return float(np.mean((r[lag:] - mu) * (r[:-lag] - mu)) / np.mean((r - mu) ** 2))


def _reference_rho_sign(returns, lag):
    """Proposition 2.5: gamma_{1,2}(m) = 2 p_{m,0} - 1."""
    s = (np.asarray(returns, dtype=float) > 0).astype(int)
    return float(2.0 * np.mean(s[lag:] == s[:-lag]) - 1.0)


def _reference_rho_magnitude(returns, lag):
    """Definition 2.4 with the k=4 ladder bucketed at the median |r|."""
    r = np.asarray(returns, dtype=float)
    median = np.median(np.abs(r))
    code = np.where(
        r > 0,
        np.where(np.abs(r) > median, 3, 2),
        np.where(np.abs(r) > median, 0, 1),
    )
    delta = (code[lag:] - code[:-lag]) % 4
    return float(np.mean(np.cos(np.pi * delta / 2.0)))


def test_sign_channel_matches_closed_form_continuation_frequency():
    """Proposition 2.5 is an identity, so it must hold exactly, not approximately."""
    import rtta

    returns = _ar1(6000, phi=-0.2, seed=1)
    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(_prices(returns))
    assert np.asarray(tail.rho_sign)[-1] == pytest.approx(
        _reference_rho_sign(returns, 1), abs=2e-3
    )


def test_channels_match_full_sample_reference():
    import rtta

    returns = _ar1(6000, phi=-0.2, seed=2)
    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(_prices(returns))
    assert np.asarray(tail.rho)[-1] == pytest.approx(_reference_rho(returns, 1), abs=2e-3)
    assert np.asarray(tail.rho_magnitude)[-1] == pytest.approx(
        _reference_rho_magnitude(returns, 1), abs=5e-3
    )


def test_fejer_identity_reproduces_the_variance_ratio():
    """Proposition 2.2: VR(q) = 1 + 2 * sum (1 - m/q) rho(m), verified against
    the variance of q-period sums rather than against itself."""
    import rtta

    returns = _ar1(8000, phi=-0.25, seed=3)
    for horizon in (2, 5, 10):
        tail = rtta.FourierResidueIdentity(horizon=horizon, max_lag=12, **{
            k: v for k, v in FULL_SAMPLE.items() if k != "max_lag"
        }).batch(_prices(returns))
        rolled = np.convolve(returns, np.ones(horizon), "valid")
        direct = rolled.var() / (horizon * returns.var())
        assert np.asarray(tail.variance_ratio)[-1] == pytest.approx(direct, abs=1e-2)


def test_negative_ar1_is_reversal_in_both_channels():
    """Genuine directional reversal must light up the sign channel."""
    import rtta

    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(
        _prices(_ar1(8000, phi=-0.25, seed=4))
    )
    assert np.asarray(tail.rho)[-1] < -0.15
    assert np.asarray(tail.rho_sign)[-1] < -0.10
    assert np.asarray(tail.z_sign)[-1] < -3.0
    assert np.asarray(tail.variance_ratio)[-1] < 1.0
    assert np.asarray(tail.variance_ratio_sign)[-1] < 1.0


def test_positive_ar1_is_momentum_in_both_channels():
    import rtta

    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(
        _prices(_ar1(8000, phi=0.25, seed=5))
    )
    assert np.asarray(tail.rho)[-1] > 0.15
    assert np.asarray(tail.rho_sign)[-1] > 0.10
    assert np.asarray(tail.variance_ratio)[-1] > 1.0
    assert np.asarray(tail.variance_ratio_sign)[-1] > 1.0


def test_iid_random_walk_leaves_every_channel_flat():
    import rtta

    rng = np.random.default_rng(6)
    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(
        _prices(rng.normal(0.0, 0.01, 8000))
    )
    assert abs(np.asarray(tail.rho)[-1]) < 0.05
    assert abs(np.asarray(tail.rho_sign)[-1]) < 0.05
    assert np.asarray(tail.variance_ratio)[-1] == pytest.approx(1.0, abs=0.1)
    assert np.asarray(tail.variance_ratio_sign)[-1] == pytest.approx(1.0, abs=0.1)


def test_roll_bounce_reverses_and_is_not_filtered_out_by_the_sign_gate():
    """A pure Roll bounce is negative-rho by construction (Eq. 7).

    It also lights up the *sign* channel, so the gate is a direction/magnitude
    separation, not a bounce filter. This is a documented limit, pinned here so
    the claim in the algorithm page and the code agree.
    """
    import rtta

    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(
        _prices(_roll_bounce(8000, half_spread=0.0045, seed=7))
    )
    assert np.asarray(tail.rho)[-1] < -0.02
    assert np.asarray(tail.rho_magnitude)[-1] < 0.0
    assert np.asarray(tail.variance_ratio)[-1] < 1.0
    # Grothendieck pins the sign channel near (2/pi) arcsin(rho) for a
    # near-Gaussian pair, so the bounce is NOT sign-neutral.
    assert np.asarray(tail.z_sign)[-1] < -2.0
    assert np.asarray(tail.elliptical_ratio)[-1] > 0.6


def test_elliptical_ratio_is_near_one_for_gaussian_linear_dependence():
    """Grothendieck's identity E[sgn X sgn Y] = (2/pi) arcsin(rho) pins the
    sign channel for an elliptical process, so the ratio sits near 1."""
    import rtta

    returns = _ar1(20000, phi=-0.2, seed=8)
    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(_prices(returns))
    assert np.asarray(tail.elliptical_ratio)[-1] == pytest.approx(1.0, abs=0.15)


def test_elliptical_ratio_collapses_when_reversal_is_magnitude_only():
    """Reversal carried only by large moves leaves typical-day direction a coin
    flip: rho stays negative while the sign channel goes quiet."""
    import rtta

    rng = np.random.default_rng(9)
    n = 20000
    returns = rng.normal(0.0, 0.005, n)
    # Overlay a reversal that only fires after a large move.
    for i in range(1, n):
        if abs(returns[i - 1]) > 0.010:
            returns[i] -= 0.55 * returns[i - 1]
    tail = rtta.FourierResidueIdentity(**FULL_SAMPLE).batch(_prices(returns))
    assert np.asarray(tail.rho)[-1] < -0.02
    assert abs(np.asarray(tail.elliptical_ratio)[-1]) < 0.6


def test_persistence_separates_sampling_noise_from_structure():
    """Definition 2.6 / Proposition 2.7: R_N -> sqrt(2) under IID noise and
    R_N -> 1 under genuine serial dependence."""
    import rtta

    rng = np.random.default_rng(10)
    noise = rtta.FourierResidueIdentity(span=512, max_lag=8).batch(
        _prices(rng.normal(0.0, 0.01, 20000))
    )
    structural = rtta.FourierResidueIdentity(span=512, max_lag=8).batch(
        _prices(_ar1(20000, phi=-0.25, seed=11))
    )
    noise_r = float(np.nanmean(np.asarray(noise.persistence)[-4000:]))
    struct_r = float(np.nanmean(np.asarray(structural.persistence)[-4000:]))
    assert struct_r < noise_r
    assert struct_r == pytest.approx(1.0, abs=0.25)
    assert noise_r == pytest.approx(math.sqrt(2.0), abs=0.35)


def test_signal_stays_flat_while_the_sign_channel_is_insignificant():
    """The practitioner payoff: no directional trade without direction evidence."""
    import rtta

    rng = np.random.default_rng(12)
    tail = rtta.FourierResidueIdentity(span=512, entry_z=3.0, exit_z=1.5).batch(
        _prices(rng.normal(0.0, 0.01, 4000))
    )
    signal = np.asarray(tail.signal)
    z_sign = np.asarray(tail.z_sign)
    assert np.all(signal[np.abs(z_sign) < 1.5] == 0.0)


def test_signal_fires_against_the_prior_move_under_strong_reversal():
    import rtta

    returns = _ar1(4000, phi=-0.35, seed=13)
    tail = rtta.FourierResidueIdentity(span=512, entry_z=2.0, exit_z=1.0).batch(
        _prices(returns)
    )
    signal = np.asarray(tail.signal)[-1000:]
    assert np.count_nonzero(signal) > 500
    # rho_sign < 0 means the stance opposes the sign of the lag-1 return.
    prior_sign = np.sign(returns[-1000:])
    active = signal != 0
    assert np.mean(signal[active] == -prior_sign[active]) > 0.9


def test_update_matches_batch_elementwise():
    import rtta

    prices = _prices(_ar1(600, phi=-0.2, seed=14))
    batched = rtta.FourierResidueIdentity(span=256).batch(prices)
    streamed = rtta.FourierResidueIdentity(span=256)
    for index, price in enumerate(prices):
        result = streamed.update(float(price))
        for field in FIELDS:
            expected = np.asarray(getattr(batched, field))[index]
            actual = getattr(result, field)
            if math.isnan(expected):
                assert math.isnan(actual)
            else:
                assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_last_and_advance_track_update():
    import rtta

    prices = _prices(_ar1(300, phi=-0.2, seed=15))
    updated = rtta.FourierResidueIdentity(span=256)
    advanced = rtta.FourierResidueIdentity(span=256)
    for price in prices:
        result = updated.update(float(price))
        advanced.advance(float(price))
        for field in FIELDS:
            expected = getattr(result, field)
            actual = getattr(advanced.last(), field)
            if math.isnan(expected):
                assert math.isnan(actual)
            else:
                assert actual == pytest.approx(expected)


def test_ohlc_overload_ignores_open_high_low():
    import rtta

    prices = _prices(_ar1(300, phi=-0.2, seed=16))
    close_only = rtta.FourierResidueIdentity(span=256)
    ohlc = rtta.FourierResidueIdentity(span=256)
    for price in prices:
        expected = close_only.update(float(price))
        actual = ohlc.update(float(price) * 0.5, float(price) * 2.0, 1.0, float(price))
        for field in FIELDS:
            if math.isnan(getattr(expected, field)):
                assert math.isnan(getattr(actual, field))
            else:
                assert getattr(actual, field) == pytest.approx(getattr(expected, field))


def test_reset_restores_a_fresh_estimator():
    import rtta

    prices = _prices(_ar1(400, phi=-0.3, seed=17))
    indicator = rtta.FourierResidueIdentity(span=256)
    for price in prices:
        indicator.update(float(price))
    indicator.reset()

    fresh = rtta.FourierResidueIdentity(span=256)
    for price in prices:
        expected = fresh.update(float(price))
        actual = indicator.update(float(price))
        for field in FIELDS:
            if math.isnan(getattr(expected, field)):
                assert math.isnan(getattr(actual, field))
            else:
                assert getattr(actual, field) == pytest.approx(getattr(expected, field))


def test_fillna_false_yields_nan_during_warmup():
    import rtta

    result = rtta.FourierResidueIdentity(max_lag=8, fillna=False).update(100.0)
    for field in FIELDS:
        assert math.isnan(getattr(result, field))


def test_fillna_true_warmup_is_neutral():
    import rtta

    result = rtta.FourierResidueIdentity(max_lag=8, fillna=True).update(100.0)
    assert result.signal == 0.0
    assert result.rho == 0.0
    assert result.variance_ratio == 1.0


def test_non_finite_and_non_positive_prices_are_rejected_without_state_damage():
    import rtta

    indicator = rtta.FourierResidueIdentity(span=256, fillna=False)
    for price in _prices(_ar1(200, phi=-0.2, seed=18)):
        indicator.update(float(price))
    healthy = indicator.last().rho

    for bad in (float("nan"), float("inf"), 0.0, -5.0):
        assert math.isnan(indicator.update(bad).rho)
    # The estimator recovers on the next valid observation.
    assert not math.isnan(indicator.update(100.0).rho)
    assert math.isfinite(healthy)


def test_horizon_and_test_lag_are_clamped_into_a_consistent_window():
    import rtta

    # max_lag must cover both horizon - 1 and test_lag; construction should not
    # read outside the ring buffer when the caller understates it.
    indicator = rtta.FourierResidueIdentity(max_lag=1, horizon=10, test_lag=6, span=256)
    for price in _prices(_ar1(500, phi=-0.2, seed=19)):
        result = indicator.update(float(price))
    assert math.isfinite(result.variance_ratio)
    assert math.isfinite(result.rho_sign)
