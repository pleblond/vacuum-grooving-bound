"""Core Protocol v3.1 pipeline.

Phase 1 Ingest  -> circulating power, cumulative energy, temperature deviation.
Phase 2 Detrend -> thermal kept as regressor (Eq 1), not pre-subtracted.
Phase 3 Regress -> per-epoch slopes on RAW frequency offsets + WLS fit.
Phase 4 Bound   -> H0: alpha = 0, 95% upper bound converted to dN/yr.

Design notes
------------
* Validation uses boolean checks (``is_valid_frame``); exceptions are only
  raised for genuinely unexpected misuse (wrong types), never for routine
  data-quality control flow.
* No ``np.unwrap`` anywhere in the frequency path: ``nu_corr_Hz`` is a
  frequency offset. Step discontinuities at unlocks/mode jumps are handled
  by epoch grouping. :func:`unwrap_beat_phase` exists solely for raw
  beat-note *phase* logs (radians) and must never touch ``nu_corr_Hz``.
* Aging: ``beta_window_edges`` fits a separate silicon-aging intercept per
  time window (paper option (a)); ``fit_quadratic_thermal`` adds the
  gamma2*dT^2 term needed at the 124 K zero-crossing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600  # 31_557_600
PI = np.pi

REQUIRED_COLUMNS = ("time_s", "nu_corr_Hz", "P_trans_W", "temp_K", "is_on")


@dataclass
class PipelineConfig:
    """Tunable inputs for :func:`run_pipeline`."""

    nu0_Hz: float = 194.4e12  # Si cavity carrier at 1542 nm
    f_nominal: float = 5e5  # nominal finesse when ring-down unavailable
    nominal_aom_Hz: float = 0.0  # subtract if logs hold absolute AOM freq
    setpoint_K: Optional[float] = None  # cryostat setpoint; median(on) if None
    min_epoch_s: float = 7200.0  # drop epochs shorter than this
    dt_s: Optional[float] = None  # sampling step; median diff if None
    use_trans_proxy: bool = False  # regress vs P_trans directly (R3 fallback)
    reference_power_W: float = 100e-6  # bound quoted "per 100 uW"
    allan_Hz: Optional[float] = None  # fixed slope weight; per-epoch SE if None
    cumulative_fit: bool = True  # also run the E_cum cross-check fit
    beta_window_edges: tuple[float, ...] = ()  # split times [s]; [] = one beta
    fit_quadratic_thermal: bool = False  # add gamma2*dT^2 (124 K crossing)


@dataclass
class EpochResult:
    t_mid_s: float
    duration_s: float
    n_samples: int
    slope_Hz_per_s: float
    slope_se_Hz_per_s: float
    p_mean_W: float
    dt_mean_K: float


@dataclass
class GroovingFit:
    beta_Si: float  # single-window beta, or the LAST window's beta if split
    alpha: float
    gamma: float
    se_beta: float
    se_alpha: float
    se_gamma: float
    r_squared: float
    n_epochs: int
    power_kind: str  # "P_circ" or "P_trans_proxy"
    beta_per_window: tuple[float, ...] = ()  # all window betas, time order
    gamma2: float = 0.0  # dT^2 coefficient (0 unless fit_quadratic_thermal)
    se_gamma2: float = 0.0


@dataclass
class CumulativeFit:
    beta_Si: float
    alpha: float
    gamma: float
    r_squared: float
    n_samples: int


@dataclass
class PipelineResult:
    epochs: list[EpochResult] = field(default_factory=list)
    grooving: Optional[GroovingFit] = None
    cumulative: Optional[CumulativeFit] = None
    # 95% upper bound on |alpha| and the derived vacuum-index drift bound.
    alpha_bound_95: float = float("nan")
    dn_per_year_bound_95: float = float("nan")
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation (boolean, no exceptions for routine quality checks)
# ---------------------------------------------------------------------------


def is_valid_frame(df: pd.DataFrame) -> bool:
    """Return True when *df* has everything the pipeline needs.

    Checks required columns, finite numerics, monotonic time, binary
    ``is_on``, and a minimum of two locked samples. Callers branch on the
    return value instead of catching exceptions.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False
    if any(c not in df.columns for c in REQUIRED_COLUMNS):
        return False
    if len(df) < 2:
        return False
    numeric = ["time_s", "nu_corr_Hz", "P_trans_W", "temp_K"]
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False
        if not bool(np.all(np.isfinite(df[col].to_numpy(dtype=float)))):
            return False
    t = df["time_s"].to_numpy(dtype=float)
    if not bool(np.all(np.diff(t) > 0)):
        return False
    is_on = df["is_on"].to_numpy()
    if not bool(np.all(np.isin(is_on, [0, 1, True, False]))):
        return False
    if int(np.sum(np.asarray(is_on, dtype=int))) < 2:
        return False
    if bool((df["P_trans_W"].to_numpy(dtype=float) < 0).any()):
        return False
    return True


def _resolve_dt(df: pd.DataFrame, cfg: PipelineConfig) -> float:
    if cfg.dt_s is not None and cfg.dt_s > 0:
        return float(cfg.dt_s)
    diffs = np.diff(df["time_s"].to_numpy(dtype=float))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))


def _resolve_setpoint(df: pd.DataFrame, cfg: PipelineConfig) -> float:
    if cfg.setpoint_K is not None:
        return float(cfg.setpoint_K)
    locked = df.loc[df["is_on"].astype(int) == 1, "temp_K"]
    if len(locked) == 0:
        return float(df["temp_K"].median())
    return float(locked.median())


# ---------------------------------------------------------------------------
# Phase 1: ingest
# ---------------------------------------------------------------------------


def _add_power_columns(df: pd.DataFrame, cfg: PipelineConfig, notes: list[str]) -> pd.DataFrame:
    """Add P_circ, E_cum, dT columns (Refinement 3 finesse handling)."""
    out = df.copy()
    on = out["is_on"].astype(int).to_numpy(dtype=float)

    if "F_measured" in out.columns and bool(out["F_measured"].notna().any()):
        finesse = out["F_measured"].interpolate(limit_direction="both").bfill().ffill()
        notes.append("Finesse: interpolated ring-down F_measured used for P_circ.")
    else:
        finesse = pd.Series(float(cfg.f_nominal), index=out.index, dtype=float)
        notes.append(
            f"Finesse: no F_measured column; nominal F={cfg.f_nominal:.3g} used "
            "(approximate; finesse drift unquantified in open data;"
            " use periodic ring-down F_measured when available)."
        )

    out["P_circ_W"] = out["P_trans_W"].to_numpy(dtype=float) * finesse.to_numpy(dtype=float) / PI * on
    dt = _resolve_dt(out, cfg)
    out["E_cum_J"] = (out["P_circ_W"].cumsum() * dt).astype(float)  # never reset
    setpoint = _resolve_setpoint(out, cfg)
    out["dT_K"] = out["temp_K"].to_numpy(dtype=float) - setpoint
    out["nu_Hz"] = out["nu_corr_Hz"].to_numpy(dtype=float) - float(cfg.nominal_aom_Hz)
    notes.append(f"Setpoint {setpoint:.4f} K; dt={dt:.4g} s; E_cum never reset.")
    return out


# ---------------------------------------------------------------------------
# Phase 3a: per-epoch slopes on RAW frequency offsets (Refinement 2)
# ---------------------------------------------------------------------------


def extract_epochs(prepared: pd.DataFrame, cfg: PipelineConfig) -> list[EpochResult]:
    """Split on lock changes and fit d(nu)/dt within each locked epoch.

    The fit uses the raw frequency offset directly. There is deliberately
    no phase-unwrap step: ``nu_corr_Hz`` is already a frequency.
    """
    on = prepared["is_on"].astype(int)
    epoch_id = (on.diff() != 0).cumsum()
    epochs: list[EpochResult] = []
    power_col = "P_trans_W" if cfg.use_trans_proxy else "P_circ_W"

    for _, ep in prepared.groupby(epoch_id):
        if int(ep["is_on"].iloc[0]) == 0:
            continue
        duration = float(ep["time_s"].iloc[-1] - ep["time_s"].iloc[0])
        if duration < float(cfg.min_epoch_s) or len(ep) < 3:
            continue
        t0 = ep["time_s"].to_numpy(dtype=float) - float(ep["time_s"].iloc[0])
        nu = ep["nu_Hz"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(t0, nu, 1)
        resid = nu - (slope * t0 + intercept)
        dof = max(len(t0) - 2, 1)
        sxx = float(np.sum((t0 - t0.mean()) ** 2))
        slope_se = float(np.sqrt(np.sum(resid**2) / dof / sxx)) if sxx > 0 else float("inf")
        epochs.append(
            EpochResult(
                t_mid_s=float(ep["time_s"].mean()),
                duration_s=duration,
                n_samples=int(len(ep)),
                slope_Hz_per_s=float(slope),
                slope_se_Hz_per_s=slope_se,
                p_mean_W=float(ep[power_col].mean()),
                dt_mean_K=float(ep["dT_K"].mean()),
            )
        )
    return epochs


# ---------------------------------------------------------------------------
# Phase 3b: WLS grooving regression  s_i = beta + alpha*P + gamma*dT
# ---------------------------------------------------------------------------


def unwrap_beat_phase(phase_rad: np.ndarray) -> np.ndarray:
    """Unwrap raw beat-note PHASE (radians) before differentiating to frequency.

    Use ONLY when re-analyzing raw beat phase vs maser/clock. NEVER apply to
    ``nu_corr_Hz``, which is already a frequency offset (R2). A missed 2 pi
    slip would inject ~100 mHz-class steps into the derived frequency.
    """
    return np.unwrap(np.asarray(phase_rad, dtype=float))


def fit_grooving_model(epochs: list[EpochResult], cfg: PipelineConfig) -> Optional[GroovingFit]:
    quad = bool(cfg.fit_quadratic_thermal)
    edges = tuple(sorted(cfg.beta_window_edges))
    t_mid = np.array([e.t_mid_s for e in epochs])
    win = np.digitize(t_mid, edges) if edges else np.zeros(len(epochs), dtype=int)
    n_win = int(win.max()) + 1 if len(epochs) else 1
    n_param = n_win + 2 + (1 if quad else 0)
    if len(epochs) < n_param + 1:  # need residual dof
        return None
    s = np.array([e.slope_Hz_per_s for e in epochs])
    p = np.array([e.p_mean_W for e in epochs])
    d = np.array([e.dt_mean_K for e in epochs])
    cols = [(win == k).astype(float) for k in range(n_win)] + [p, d]
    if quad:
        cols.append(d**2)
    X = np.column_stack(cols)

    if cfg.allan_Hz is not None and cfg.allan_Hz > 0:
        w = np.full_like(s, 1.0 / cfg.allan_Hz**2)
    else:
        se = np.array([e.slope_se_Hz_per_s for e in epochs])
        se = np.where(np.isfinite(se) & (se > 0), se, np.median(se[np.isfinite(se) & (se > 0)]))
        w = 1.0 / se**2
    sqrt_w = np.sqrt(w)
    Xw, sw = X * sqrt_w[:, None], s * sqrt_w
    coeffs, _, rank, _ = np.linalg.lstsq(Xw, sw, rcond=None)
    if rank < n_param:
        return None
    resid = sw - Xw @ coeffs
    dof = max(len(s) - n_param, 1)
    mse = float(resid @ resid / dof)
    try:
        cov = mse * np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        return None
    se_vec = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ss_tot = float(np.sum((sw - sw.mean()) ** 2))
    r2 = float(1.0 - resid @ resid / ss_tot) if ss_tot > 0 else 0.0
    betas = tuple(float(c) for c in coeffs[:n_win])
    return GroovingFit(
        beta_Si=float(coeffs[n_win - 1]),
        alpha=float(coeffs[n_win]),
        gamma=float(coeffs[n_win + 1]),
        se_beta=float(se_vec[n_win - 1]),
        se_alpha=float(se_vec[n_win]),
        se_gamma=float(se_vec[n_win + 1]),
        r_squared=r2,
        n_epochs=len(epochs),
        power_kind="P_trans_proxy" if cfg.use_trans_proxy else "P_circ",
        beta_per_window=betas,
        gamma2=float(coeffs[n_win + 2]) if quad else 0.0,
        se_gamma2=float(se_vec[n_win + 2]) if quad else 0.0,
    )


# ---------------------------------------------------------------------------
# Phase 3c: cumulative cross-check fit
# ---------------------------------------------------------------------------


def fit_cumulative_model(prepared: pd.DataFrame, cfg: PipelineConfig) -> Optional[CumulativeFit]:
    """Fit nu(t) = b*t + a*E_cum + g*cum_dT + epoch steps + const.

    The step dummies absorb unlock/mode-jump offsets so the cumulative
    slope cannot be faked by a relock offset. Returns None when the design
    is rank-deficient.
    """
    locked = prepared.loc[prepared["is_on"].astype(int) == 1].copy()
    if len(locked) < 10:
        return None
    dt = _resolve_dt(prepared, cfg)
    t = locked["time_s"].to_numpy(dtype=float) - float(locked["time_s"].iloc[0])
    y = locked["nu_Hz"].to_numpy(dtype=float)
    e = locked["E_cum_J"].to_numpy(dtype=float)
    cum_dt = (locked["dT_K"].cumsum() * dt).to_numpy(dtype=float)

    on = prepared["is_on"].astype(int)
    epoch_id = (on.diff() != 0).cumsum()
    locked_ids = epoch_id.loc[locked.index].to_numpy()
    _, inv = np.unique(locked_ids, return_inverse=True)
    steps = np.zeros((len(locked), int(inv.max())), dtype=float)
    for k in range(int(inv.max())):  # one dummy per relock after the first
        steps[:, k] = (inv > k).astype(float)

    power_col = "P_trans_W" if cfg.use_trans_proxy else "P_circ_W"
    _ = power_col  # cumulative form always integrates P_circ into E_cum
    X = np.column_stack([t, e, cum_dt, steps, np.ones_like(t)])
    coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < 3:
        return None
    pred = X @ coeffs
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return CumulativeFit(
        beta_Si=float(coeffs[0]),
        alpha=float(coeffs[1]),
        gamma=float(coeffs[2]),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        n_samples=int(len(locked)),
    )


# ---------------------------------------------------------------------------
# Phase 4: bound
# ---------------------------------------------------------------------------


def bound_from_fit(fit: GroovingFit, cfg: PipelineConfig) -> tuple[float, float]:
    """95% upper bound on |alpha| and the derived Δn/yr per reference power.

    s = -nu0 * d(Δn)/dt for the power-dependent part, so
    Δn/yr = |alpha| * P_ref * sec/yr / nu0.
    """
    alpha_up = abs(fit.alpha) + 1.96 * fit.se_alpha
    dn = alpha_up * float(cfg.reference_power_W) * SECONDS_PER_YEAR / float(cfg.nu0_Hz)
    return float(alpha_up), float(dn)


def run_pipeline(df: pd.DataFrame, cfg: Optional[PipelineConfig] = None) -> PipelineResult:
    """Run Phases 1-4. Returns a result with notes; empty epochs when invalid."""
    cfg = cfg or PipelineConfig()
    res = PipelineResult()
    if not is_valid_frame(df):
        res.notes.append("Input failed validation (columns/finiteness/monotonic time/lock). No fit.")
        return res
    prepared = _add_power_columns(df, cfg, res.notes)
    res.notes.append("Thermal kept as gamma*dT regressor (R1); no pre-subtraction.")
    res.notes.append("Slopes fit on raw nu_corr_Hz offsets; no phase unwrap (R2).")
    res.epochs = extract_epochs(prepared, cfg)
    if len(res.epochs) == 0:
        res.notes.append("No locked epochs survived the duration cut. No fit.")
        return res
    if cfg.beta_window_edges:
        res.notes.append(f"Windowed aging: beta split at {sorted(cfg.beta_window_edges)} s.")
    if cfg.fit_quadratic_thermal:
        res.notes.append("Quadratic thermal gamma2*dT^2 included (124 K crossing).")
    res.grooving = fit_grooving_model(res.epochs, cfg)
    if res.grooving is None:
        res.notes.append("Grooving regression rank-deficient or <4 epochs. No bound.")
        return res
    res.alpha_bound_95, res.dn_per_year_bound_95 = bound_from_fit(res.grooving, cfg)
    if cfg.cumulative_fit:
        res.cumulative = fit_cumulative_model(prepared, cfg)
        if res.cumulative is not None:
            res.notes.append(
                f"Cumulative cross-check: alpha_E={res.cumulative.alpha:.3g} Hz/s/W "
                f"(R2={res.cumulative.r_squared:.3f})."
            )
    res.notes.append(
        f"Bound: |alpha|<= {res.alpha_bound_95:.3g} Hz/s/W (95%), "
        f"dn/yr <= {res.dn_per_year_bound_95:.3g} per {cfg.reference_power_W*1e6:.0f} uW."
    )
    return res
