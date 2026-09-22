"""Synthetic cavity-log generator for protocol validation and demos.

Produces the exact columns the pipeline ingests so labs can check out the
regression on data with a known injected grooving coefficient ``alpha``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(
    n_epochs: int = 8,
    epoch_s: float = 12_000.0,
    gap_s: float = 3_000.0,
    dt_s: float = 10.0,
    nu0_Hz: float = 194.4e12,
    f_finesse: float = 5e5,
    beta_Si: float = 1e-6,
    beta_1: float = 0.0,
    beta_tau_s: float = 0.0,
    alpha: float = 0.0,
    gamma: float = 20.0,
    gamma2: float = 0.0,
    p_trans_levels_W: tuple[float, ...] = (0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6),
    temp_noise_mK: float = 0.05,
    freq_noise_Hz: float = 2.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic locked/gap log with known (beta, alpha, gamma)."""
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    t = 0.0
    nu_offset = 0.0
    p_circ_scale = f_finesse / np.pi

    for k in range(n_epochs):
        p_trans = float(p_trans_levels_W[k % len(p_trans_levels_W)])
        p_circ = p_trans * p_circ_scale
        n = int(epoch_s / dt_s)
        tt = t + np.arange(n) * dt_s
        # Per-epoch quasi-static temperature offset (Protocol v3.1 Eq 1
        # assumes dT is ~constant within an epoch at some lock-point offset
        # and varies between epochs; a small within-epoch trend + readout
        # noise ride on top as unmodeled realism).
        offset_K = rng.uniform(-1, 1) * 1e-3 * temp_noise_mK
        frac = np.arange(n) / n
        trend_K = 0.02 * offset_K * (frac - 0.5)
        meas_noise_K = rng.standard_normal(n) * 1e-3 * temp_noise_mK * 0.02
        temp_wander_K = offset_K + trend_K + meas_noise_K
        true_wander_K = offset_K + trend_K  # what the cavity actually feels
        steps = np.arange(n) * dt_s
        beta_now = beta_Si + (
            beta_1 * np.exp(-t / beta_tau_s) if beta_tau_s > 0 else 0.0
        )
        # Model-consistent signal: (beta + alpha*P) * t + gamma * integral(dT).
        # The per-epoch slope regression then sees gamma * mean(dT) to first
        # order, while the cumulative fit sees the exact integral.
        nu = (
            nu_offset
            + (beta_now + alpha * p_circ + gamma2 * offset_K**2) * steps
            + gamma * np.cumsum(true_wander_K) * dt_s
            + rng.standard_normal(n) * freq_noise_Hz
        )
        rows.append(
            pd.DataFrame(
                {
                    "time_s": tt,
                    "nu_corr_Hz": nu,
                    "P_trans_W": np.full(n, p_trans),
                    "temp_K": 4.0 + temp_wander_K,
                    "is_on": np.ones(n, dtype=int),
                }
            )
        )
        nu_offset = float(nu[-1] + rng.standard_normal() * 50.0)  # relock jump
        # unlocked gap (flatlines E_cum contribution via is_on=0)
        m = max(int(gap_s / dt_s), 1)
        tg = t + epoch_s + np.arange(m) * dt_s
        rows.append(
            pd.DataFrame(
                {
                    "time_s": tg,
                    "nu_corr_Hz": np.full(m, nu_offset),
                    "P_trans_W": np.zeros(m),
                    "temp_K": np.full(m, 4.0),
                    "is_on": np.zeros(m, dtype=int),
                }
            )
        )
        t = float(tg[-1] + dt_s)

    df = pd.concat(rows, ignore_index=True)
    _ = nu0_Hz  # carrier documented for bound conversion downstream
    return df
