"""vacuum_creep: bounding DC vacuum index creep from cryogenic Si cavity drift logs.

Implements Protocol v3.1 (Thermal + Unwrap + Finesse corrections):

    s_i = beta_Si + alpha * <P>_i + gamma * dT_i + eps_i        (Eq 1)

where s_i is the per-epoch AOM correction drift rate [Hz/s],
<P>_i the mean circulating (or transmitted-proxy) power in epoch i,
and dT_i the mean cryostat temperature deviation in epoch i.

Key v3.1 corrections vs v3.0:
  R1 (thermal): gamma*dT regressor kept even at the 4 K / 124 K
      zero-crossing; never pre-subtracted so covariance is preserved.
  R2 (unwrap): nu_corr is a frequency offset in Hz, NOT wrapped phase.
      Slopes are fit on the raw offset; no np.unwrap is applied.
  R3 (finesse): P_circ = P_trans * F / pi with time-interpolated
      measured finesse when available, else nominal F with a
      documented <~2% conservative error, or direct P_trans proxy mode.
"""

from vacuum_creep.pipeline import (
    PipelineConfig,
    EpochResult,
    GroovingFit,
    PipelineResult,
    is_valid_frame,
    run_pipeline,
    unwrap_beat_phase,
)

__all__ = [
    "PipelineConfig",
    "EpochResult",
    "GroovingFit",
    "PipelineResult",
    "is_valid_frame",
    "run_pipeline",
    "unwrap_beat_phase",
]

__version__ = "3.1.0"
