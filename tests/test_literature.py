import numpy as np
import pandas as pd

from vacuum_creep.literature import (
    NU0_HZ,
    PAPER_ANCHORS,
    SI6_P_CIRC_W,
    constant_power_sensitivity,
    frac_to_hz_per_s,
    has_digitized_data,
    hz_per_s_to_frac,
    integrate_drift,
    lengthening_m,
    load_digitized,
    p_circ,
    settled_stats,
)


def test_frac_conversions_match_paper():
    for cav, a in PAPER_ANCHORS.items():
        got = hz_per_s_to_frac(a["recent_uHz_s"] * 1e-6)
        assert abs(got - a["recent_frac_s"]) / abs(a["recent_frac_s"]) < 0.03, cav
    assert abs(frac_to_hz_per_s(-2.6e-19) - -50e-6) < 2e-6


def test_44khz_is_48pm():
    assert abs(lengthening_m(-44e3, 0.21) - 48e-12) / 48e-12 < 0.05


def test_si6_circulating_power():
    assert abs(p_circ(90e-9, 470_000) - SI6_P_CIRC_W) < 1e-9
    assert abs(SI6_P_CIRC_W - 0.0135) / 0.0135 < 0.02


def test_digitized_data_present_and_shaped():
    assert has_digitized_data()
    assert not has_digitized_data("/nonexistent/path.csv")
    df = load_digitized()
    assert df is not None
    assert set(df["cavity"].unique()) == {"Si2", "Si3", "Si5", "Si6"}
    assert (df["n_pixels"] >= 8).all()


def test_digitized_endpoints_match_paper_order_of_magnitude():
    df = load_digitized()
    for cav, a in PAPER_ANCHORS.items():
        last = df.loc[df["cavity"] == cav].iloc[-1]["drift_rate_uHz_s"]
        assert np.sign(last) == np.sign(a["recent_uHz_s"]), cav
        ratio = abs(last / a["recent_uHz_s"])
        assert 0.25 < ratio < 4.0, (cav, last)


def test_si2_integral_cross_check():
    # Paper: -44 kHz over 10 yr; digitized span starts day 260, so the
    # integral must be negative and somewhat smaller in magnitude.
    df = load_digitized()
    tot = integrate_drift(df, "Si2")
    assert -60e3 < tot < -25e3


def test_si6_integral_positive_small():
    df = load_digitized()
    tot = integrate_drift(df, "Si6")
    assert 0 < tot < 20e3


def test_settled_eras_bracket_paper_values():
    df = load_digitized()
    for cav, day0 in (("Si2", 2500), ("Si3", 2500), ("Si5", 2000), ("Si6", 400)):
        med, std, n = settled_stats(df, cav, day0)
        assert n > 3
        assert abs(med - PAPER_ANCHORS[cav]["recent_uHz_s"]) < 4 * max(abs(std), 1.0), (cav, med)


def test_constant_power_sensitivity_math():
    alpha, dn = constant_power_sensitivity(10e-6, 0.01)
    assert alpha == 1e-3
    assert abs(dn - 1e-3 * 100e-6 * 365.25 * 24 * 3600 / NU0_HZ) / dn < 1e-9


def test_sensitivity_scales_with_excursion():
    a1, _ = constant_power_sensitivity(1e-5, 0.001)
    a2, _ = constant_power_sensitivity(1e-5, 0.01)
    assert a1 == 10 * a2


def test_load_missing_returns_none():
    assert load_digitized("/nonexistent/path.csv") is None
    empty = pd.DataFrame(columns=["cavity", "days_since_contacting", "drift_rate_uHz_s"])
    med, std, n = settled_stats(empty, "Si2", 0.0)
    assert n == 0 and np.isnan(med)


def test_windowed_stats_early_late_split():
    from vacuum_creep.literature import load_digitized, windowed_stats

    df = load_digitized()
    w = windowed_stats(df, "Si2", 2000.0)
    assert w["early"][2] == 42 and w["late"][2] == 13
    assert w["late"][1] < w["full"][1]  # aging bias removed -> tighter scatter
    assert abs(w["late"][0] - -59.0) < 5.0


def test_aging_exp_beats_1overT():
    from vacuum_creep.literature import aging_fit_1overT, aging_fit_exp, load_digitized

    df = load_digitized()
    b0t, _, rst, _ = aging_fit_1overT(df, "Si2", 800.0)
    b0e, _, taue, rse, n = aging_fit_exp(df, "Si2", 800.0)
    assert n == 32
    assert rse < rst
    assert -70.0 < b0e < -40.0  # sane asymptote at settled level
    assert 300.0 < taue < 1500.0  # ~1.8 yr, not the draft's unsupported 3 yr
    assert abs(b0t - -57.0) > abs(b0e - -57.0)  # 1/t asymptote biased


def test_no_tick_label_outlier_bins():
    from vacuum_creep.literature import load_digitized

    df = load_digitized()
    assert df.loc[df["cavity"] == "Si2", "drift_rate_uHz_s"].min() > -600.0


def test_dither_sensitivity_math():
    import numpy as np

    from vacuum_creep.literature import NU0_HZ, SECONDS_PER_YEAR, dither_sensitivity

    a1, dn1 = dither_sensitivity(1.0, 0.0135, 1)
    assert abs(a1 - np.sqrt(2) * 1e-6 / 0.0135) / a1 < 1e-9
    assert abs(dn1 - a1 * 100e-6 * SECONDS_PER_YEAR / NU0_HZ) / dn1 < 1e-9
    a4, _ = dither_sensitivity(1.0, 0.0135, 4)
    assert abs(a4 - a1 / 2) / a4 < 1e-9


def test_allan_floor_negligible():
    from vacuum_creep.literature import slope_precision_allan

    assert slope_precision_allan() < 1e-9  # Hz/s: systematics dominate dither
