"""Lee et al. 2026 (PRL 136, 033801; arXiv:2509.13503v1) drift-record analysis.

Paper-text anchors (exact quotes from the published text, used as given):
  Si2: 21 cm, 124 K, SiO2/Ta2O5; recent -50 uHz/s (-2.6e-19 frac);
       -44 kHz cumulative over 10 yr (= +48 pm lengthening).
  Si3: 21 cm, 124 K, SiO2/Ta2O5; recent -15 uHz/s (-7.7e-20).
  Si5: 21 cm, 124 K, GaAs/AlGaAs; recent -20 uHz/s (-1.0e-19).
  Si6: 6 cm, 17 K, GaAs/AlGaAs; finesse 470000 at 1542 nm;
       90 nW cavity transmission; recent +10 uHz/s (+5.2e-20).

The full time series (Fig. 3b) is digitized in data/lee2026_fig3b.csv by
scripts/digitize_fig3b.py. There is no machine-readable supplement in the
arXiv v1 source package (three figure images only), so figure digitization
is currently the only open-data route; per-point error is ~15%.

Key honesty note for the constant-power analysis: with power held constant,
alpha*P is degenerate with beta_Si, so no alpha fit is possible. What the
published record supports today is (a) drift levels and decay shapes,
(b) the Si2 cumulative integral cross-check, and (c) sensitivity
projections: what power-excursion data from the labs would buy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

C_M_S = 299_792_458.0
LAMBDA_M = 1542e-9
NU0_HZ = C_M_S / LAMBDA_M  # ~1.9444e14 Hz
SECONDS_PER_YEAR = 365.25 * 24 * 3600

SI6_FINESSE = 470_000.0
SI6_P_TRANS_W = 90e-9
SI6_P_CIRC_W = SI6_P_TRANS_W * SI6_FINESSE / np.pi  # ~13.5 mW

REFERENCE_POWER_W = 100e-6  # bounds quoted "per 100 uW"

PAPER_ANCHORS = {
    # recent drift rates in uHz/s, fractional rates, cavity metadata
    "Si2": {"recent_uHz_s": -50.0, "recent_frac_s": -2.6e-19, "length_m": 0.21,
            "temp_K": 124.0, "mirrors": "SiO2/Ta2O5", "cumulative_kHz": -44.0,
            "cumulative_span_yr": 10.0, "cumulative_pm": 48.0},
    "Si3": {"recent_uHz_s": -15.0, "recent_frac_s": -7.7e-20, "length_m": 0.21,
            "temp_K": 124.0, "mirrors": "SiO2/Ta2O5"},
    "Si5": {"recent_uHz_s": -20.0, "recent_frac_s": -1.0e-19, "length_m": 0.21,
            "temp_K": 124.0, "mirrors": "GaAs/AlGaAs"},
    "Si6": {"recent_uHz_s": 10.0, "recent_frac_s": 5.2e-20, "length_m": 0.06,
            "temp_K": 17.0, "mirrors": "GaAs/AlGaAs", "finesse": SI6_FINESSE,
            "p_trans_W": SI6_P_TRANS_W, "p_circ_W": SI6_P_CIRC_W},
}

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lee2026_fig3b.csv"


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def hz_per_s_to_frac(s_hz_s: float) -> float:
    """Drift rate [Hz/s] -> fractional frequency drift [1/s]."""
    return float(s_hz_s) / NU0_HZ


def frac_to_hz_per_s(y: float) -> float:
    """Fractional drift [1/s] -> drift rate [Hz/s]."""
    return float(y) * NU0_HZ


def p_circ(p_trans_W: float, finesse: float) -> float:
    """Circulating power from transmitted power and finesse (R3)."""
    return float(p_trans_W) * float(finesse) / np.pi


def lengthening_m(dnu_hz: float, length_m: float) -> float:
    """Cavity length change [m] for an absolute frequency shift [Hz].

    nu = m c / 2L so dL = -L dnu / nu0; a negative frequency drift is a
    lengthening (positive return).
    """
    return -float(length_m) * float(dnu_hz) / NU0_HZ


# ---------------------------------------------------------------------------
# Digitized record
# ---------------------------------------------------------------------------


def has_digitized_data(path: str | Path = DATA_PATH) -> bool:
    """Boolean check for the digitized Fig. 3b CSV (no exceptions)."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        df = pd.read_csv(p, nrows=3)
    except (OSError, pd.errors.ParserError):
        return False
    return {"cavity", "days_since_contacting", "drift_rate_uHz_s"} <= set(df.columns)


def load_digitized(path: str | Path = DATA_PATH) -> pd.DataFrame | None:
    """Load the digitized record, or None when unavailable."""
    if not has_digitized_data(path):
        return None
    df = pd.read_csv(path)
    return df.sort_values(["cavity", "days_since_contacting"]).reset_index(drop=True)


def integrate_drift(df: pd.DataFrame, cavity: str) -> float:
    """Total absolute frequency shift [Hz] over a cavity's digitized span."""
    sub = df.loc[df["cavity"] == cavity].sort_values("days_since_contacting")
    t = sub["days_since_contacting"].to_numpy(float) * 86400.0
    r = sub["drift_rate_uHz_s"].to_numpy(float) * 1e-6
    return float(np.trapezoid(r, t))


def windowed_stats(
    df: pd.DataFrame, cavity: str, split_days: float
) -> dict[str, tuple[float, float, int]]:
    """Paper option (a): split a cavity record into early/late beta windows.

    Returns median/std/count of drift [uHz/s] for 'full', 'early'
    (days < split) and 'late' (days >= split). Separating the fast-aging
    era from the settled era removes aging bias from the late-window bound.
    """
    sub = df.loc[df["cavity"] == cavity].sort_values("days_since_contacting")
    out: dict[str, tuple[float, float, int]] = {}
    for key, part in (
        ("full", sub),
        ("early", sub.loc[sub["days_since_contacting"] < split_days]),
        ("late", sub.loc[sub["days_since_contacting"] >= split_days]),
    ):
        v = part["drift_rate_uHz_s"].to_numpy(float)
        if len(v) == 0:
            out[key] = (float("nan"), float("nan"), 0)
        else:
            out[key] = (
                float(np.median(v)),
                float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                int(len(v)),
            )
    return out


def settled_stats(df: pd.DataFrame, cavity: str, days_min: float) -> tuple[float, float, int]:
    """Median / std / count of drift [uHz/s] in the settled era (days >= min)."""
    sub = df.loc[(df["cavity"] == cavity) & (df["days_since_contacting"] >= days_min),
                 "drift_rate_uHz_s"].to_numpy(float)
    if len(sub) == 0:
        return float("nan"), float("nan"), 0
    return float(np.median(sub)), float(np.std(sub, ddof=1)) if len(sub) > 1 else 0.0, int(len(sub))


# ---------------------------------------------------------------------------
# Option (b): parametric aging model  s(t) = beta0 + beta1 / t
# ---------------------------------------------------------------------------


def aging_fit_1overT(
    df: pd.DataFrame, cavity: str, days_min: float = 0.0
) -> tuple[float, float, float, int]:
    """Fit drift [uHz/s] vs age with s(t) = beta0 + beta1/t.

    Returns (beta0, beta1, residual_std, n). Kept for comparison; on Si2
    the exponential model fits markedly better (late decay is faster
    than 1/t, so 1/t biases the asymptote high).
    """
    sub = df.loc[
        (df["cavity"] == cavity) & (df["days_since_contacting"] >= days_min)
    ].sort_values("days_since_contacting")
    t = sub["days_since_contacting"].to_numpy(float)
    r = sub["drift_rate_uHz_s"].to_numpy(float)
    X = np.column_stack([np.ones_like(t), 1.0 / t])
    coeffs, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    resid = r - X @ coeffs
    dof = max(len(t) - 2, 1)
    return float(coeffs[0]), float(coeffs[1]), float(np.sqrt(resid @ resid / dof)), int(len(t))


def aging_fit_exp(
    df: pd.DataFrame,
    cavity: str,
    days_min: float = 0.0,
    tau_grid: tuple[float, float, int] = (100.0, 10000.0, 60),
) -> tuple[float, float, float, float, int]:
    """Fit drift [uHz/s] vs age with s(t) = beta0 + beta1*exp(-t/tau).

    Tau is found by log-grid search + least squares (no extra dependency).
    Returns (beta0, beta1, tau_days, residual_std, n). For Si2 over
    days >= 800 this gives tau ~= 700 d (~1.9 yr) with residuals ~= 12,
    far better than 1/t: the late decay is faster than 1/t.
    """
    sub = df.loc[
        (df["cavity"] == cavity) & (df["days_since_contacting"] >= days_min)
    ].sort_values("days_since_contacting")
    t = sub["days_since_contacting"].to_numpy(float)
    r = sub["drift_rate_uHz_s"].to_numpy(float)
    best: tuple[float, float, float, float] | None = None
    for tau in np.logspace(np.log10(tau_grid[0]), np.log10(tau_grid[1]), tau_grid[2]):
        X = np.column_stack([np.ones_like(t), np.exp(-t / tau)])
        coeffs, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
        ss = float(((r - X @ coeffs) ** 2).sum())
        if best is None or ss < best[0]:
            best = (ss, float(coeffs[0]), float(coeffs[1]), float(tau))
    assert best is not None
    ss, b0, b1, tau = best
    dof = max(len(t) - 3, 1)
    return b0, b1, tau, float(np.sqrt(ss / dof)), int(len(t))


# ---------------------------------------------------------------------------
# Dither projection from first principles
# ---------------------------------------------------------------------------


def slope_precision_allan(
    mod_allan: float = 2.5e-17,
    tau_ref_s: float = 10.0,
    duration_s: float = 30 * 86400,
    nu0_hz: float = NU0_HZ,
) -> float:
    """Statistical slope uncertainty [Hz/s] for a drift fit over duration_s.

    White-FM model anchored on the published modified Allan deviation:
    ADEV(tau_ref) ~= MDEV * sqrt(2), extrapolated as 1/sqrt(tau), then
    slope SE = sigma_nu * sqrt(12/N) / T for N 1-s samples over T.
    This is the measurement floor; real dither runs are limited by beta
    wander and thermal systematics (see dither_sensitivity).
    """
    adev_ref = mod_allan * np.sqrt(2.0)
    sigma_nu_1s = adev_ref * np.sqrt(tau_ref_s) * nu0_hz
    n = max(duration_s, 1.0)
    return float(sigma_nu_1s * np.sqrt(12.0 / n) / duration_s)


def dither_sensitivity(
    sigma_month_uHz_s: float,
    delta_p_W: float,
    n_pairs: int = 1,
    p_ref_w: float = REFERENCE_POWER_W,
    nu0_hz: float = NU0_HZ,
) -> tuple[float, float]:
    """Alpha/dn-yr reach of an alternating power dither (synchronous demod).

    Each on/off month-pair measures a slope difference with uncertainty
    sqrt(2)*sigma_month; common drift beta cancels in the difference.
    sigma_alpha = sqrt(2)*sigma_month / (deltaP * sqrt(n_pairs)),
    converted to dn/yr per p_ref_w. sigma_month is the per-month slope
    repeatability INCLUDING beta wander and thermal systematics, not the
    Allan floor (typically ~1 uHz/s optimistic, ~5-10 realistic for Si6).
    """
    sigma_alpha = (
        np.sqrt(2.0) * abs(float(sigma_month_uHz_s)) * 1e-6
        / (abs(float(delta_p_W)) * np.sqrt(max(int(n_pairs), 1)))
    )
    dn_yr = sigma_alpha * float(p_ref_w) * SECONDS_PER_YEAR / float(nu0_hz)
    return float(sigma_alpha), float(dn_yr)


# ---------------------------------------------------------------------------
# Constant-power sensitivity projection
# ---------------------------------------------------------------------------


def constant_power_sensitivity(
    sigma_s_hz_s: float,
    delta_p_w: float,
    p_ref_w: float = REFERENCE_POWER_W,
) -> tuple[float, float]:
    """Upper limit on |alpha| given drift scatter and a power excursion.

    With constant power, alpha is degenerate with beta_Si; this answers the
    next-best question: if the power had varied by delta_p_W across epochs
    while the drift scattered by sigma_s_Hz_s, then |alpha| <= sigma/deltaP,
    converted to a dn/yr bound per p_ref_w:

        dn/yr = |alpha| * p_ref * sec/yr / nu0.
    """
    alpha_lim = abs(float(sigma_s_hz_s)) / abs(float(delta_p_w))
    dn_yr = alpha_lim * float(p_ref_w) * SECONDS_PER_YEAR / NU0_HZ
    return alpha_lim, dn_yr
