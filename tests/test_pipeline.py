import numpy as np
import pandas as pd

from vacuum_creep.pipeline import PipelineConfig, is_valid_frame, run_pipeline
from vacuum_creep.synthetic import make_synthetic


def test_valid_frame_accepts_synthetic():
    df = make_synthetic(n_epochs=2)
    assert is_valid_frame(df)


def test_valid_frame_rejects_bad_inputs():
    assert not is_valid_frame(pd.DataFrame({"a": [1]}))
    df = make_synthetic(n_epochs=1)
    bad = df.copy()
    bad.loc[5, "nu_corr_Hz"] = np.nan
    assert not is_valid_frame(bad)
    backwards = df.copy()
    backwards.loc[10, "time_s"] = backwards.loc[0, "time_s"]
    assert not is_valid_frame(backwards)
    empty = df.iloc[:1].copy()
    assert not is_valid_frame(empty)


def test_null_grooving_gives_tight_bound():
    df = make_synthetic(n_epochs=8, alpha=0.0, seed=11)
    res = run_pipeline(df, PipelineConfig())
    assert res.grooving is not None
    assert abs(res.grooving.alpha) < 5 * res.grooving.se_alpha
    assert np.isfinite(res.dn_per_year_bound_95)
    assert res.cumulative is not None


def test_injected_grooving_is_recovered():
    # inject a clearly detectable alpha; P_circ ~ 1e-6 * 5e5/pi ~ 0.16 W
    df = make_synthetic(n_epochs=10, alpha=3e-4, seed=3)
    res = run_pipeline(df, PipelineConfig())
    assert res.grooving is not None
    assert res.grooving.alpha > 0
    assert abs(res.grooving.alpha - 3e-4) / 3e-4 < 0.35


def test_thermal_term_deconfounds_power():
    df = make_synthetic(n_epochs=10, alpha=0.0, gamma=400.0, temp_noise_mK=0.3, seed=5)
    res = run_pipeline(df, PipelineConfig())
    assert res.grooving is not None
    # thermal must be absorbed by gamma, not leak into alpha
    assert abs(res.grooving.alpha) < 5 * res.grooving.se_alpha
    assert res.grooving.gamma != 0.0


def test_trans_proxy_and_finesse_modes_agree_in_sign():
    df = make_synthetic(n_epochs=8, alpha=2e-4, seed=9)
    df["F_measured"] = 5e5 * (1 + 0.005 * np.sin(np.linspace(0, 6, len(df))))
    full = run_pipeline(df, PipelineConfig(use_trans_proxy=False))
    proxy = run_pipeline(df, PipelineConfig(use_trans_proxy=True))
    assert full.grooving is not None and proxy.grooving is not None
    assert np.sign(full.grooving.alpha) == np.sign(proxy.grooving.alpha)


def test_short_epochs_yield_no_fit_not_crash():
    df = make_synthetic(n_epochs=2, epoch_s=600.0, gap_s=600.0, seed=1)
    res = run_pipeline(df, PipelineConfig(min_epoch_s=7200.0))
    assert len(res.epochs) == 0
    assert res.grooving is None


def test_no_unwrap_applied_to_frequency_steps():
    # a relock step must not bleed across epochs: exactly 2 surviving epochs
    df = make_synthetic(n_epochs=2, seed=2)
    res = run_pipeline(df, PipelineConfig(min_epoch_s=3600.0))
    assert len(res.epochs) == 2


def test_windowed_beta_removes_aging_bias():
    import pandas as pd

    from vacuum_creep.pipeline import PipelineConfig, run_pipeline
    from vacuum_creep.synthetic import make_synthetic

    early = make_synthetic(n_epochs=6, alpha=3e-4, beta_Si=5e-4,
                           p_trans_levels_W=(0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6), seed=21)
    late = make_synthetic(n_epochs=6, alpha=3e-4, beta_Si=1e-4,
                          p_trans_levels_W=(1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6), seed=22)
    late = late.copy()
    late["time_s"] = late["time_s"] + 1e7  # power rises as aging decays
    df = pd.concat([early, late], ignore_index=True)

    single = run_pipeline(df, PipelineConfig()).grooving
    split = run_pipeline(df, PipelineConfig(beta_window_edges=(5e6,))).grooving
    assert single is not None and split is not None
    assert len(split.beta_per_window) == 2
    assert split.beta_per_window[0] > split.beta_per_window[1]  # decay captured
    assert abs(split.alpha - 3e-4) < abs(single.alpha - 3e-4)
    assert abs(split.alpha - 3e-4) / 3e-4 < 0.30


def test_quadratic_thermal_recovery():
    from vacuum_creep.pipeline import PipelineConfig, run_pipeline
    from vacuum_creep.synthetic import make_synthetic

    df = make_synthetic(n_epochs=12, alpha=0.0, gamma=0.0, gamma2=1e3,
                        temp_noise_mK=0.3, seed=13)
    res = run_pipeline(df, PipelineConfig(fit_quadratic_thermal=True))
    assert res.grooving is not None
    assert abs(res.grooving.gamma2 - 1e3) / 1e3 < 0.30
    assert abs(res.grooving.alpha) < 3 * res.grooving.se_alpha


def test_unwrap_beat_phase_only_for_phase():
    import numpy as np

    from vacuum_creep import unwrap_beat_phase

    true = np.linspace(0, 20 * np.pi, 500)
    wrapped = (true + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(np.diff(unwrap_beat_phase(wrapped)), np.diff(true))


def test_windowed_fit_needs_residual_dof():
    from vacuum_creep.pipeline import PipelineConfig, run_pipeline
    from vacuum_creep.synthetic import make_synthetic

    df = make_synthetic(n_epochs=4, seed=4)
    res = run_pipeline(df, PipelineConfig(beta_window_edges=(1e4, 2e4, 3e4)))
    assert res.grooving is None  # 4 windows + P + T > epochs: fail safe
