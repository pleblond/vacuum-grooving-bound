"""CLI: run Protocol v3.1 on a CSV log or on synthetic demo data."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from vacuum_creep.pipeline import PipelineConfig, is_valid_frame, run_pipeline
from vacuum_creep.synthetic import make_synthetic


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Protocol v3.1 cavity-drift grooving bound")
    p.add_argument("--input", help="CSV with time_s,nu_corr_Hz,P_trans_W,temp_K,is_on (+optional F_measured)")
    p.add_argument("--synthetic", action="store_true", help="run on synthetic demo data (no input needed)")
    p.add_argument("--alpha-inject", type=float, default=0.0, help="injected alpha for --synthetic only")
    p.add_argument("--output", help="write JSON result to this path")
    p.add_argument("--trans-proxy", action="store_true", help="regress vs P_trans directly (R3 fallback)")
    p.add_argument("--min-epoch-s", type=float, default=7200.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        df = make_synthetic(alpha=args.alpha_inject)
        print(f"[demo] synthetic log: {len(df)} rows, injected alpha={args.alpha_inject:.3g} Hz/s/W")
    elif args.input:
        df = pd.read_csv(args.input)
    else:
        print("Provide --input CSV or --synthetic.", file=sys.stderr)
        return 2

    if not is_valid_frame(df):
        print("Input failed validation: check columns, finite numerics, monotonic time, is_on.", file=sys.stderr)
        return 1

    cfg = PipelineConfig(use_trans_proxy=args.trans_proxy, min_epoch_s=args.min_epoch_s)
    res = run_pipeline(df, cfg)
    payload = {
        "n_epochs": len(res.epochs),
        "grooving": None if res.grooving is None else {
            "beta_Si_Hz_per_s": res.grooving.beta_Si,
            "alpha_Hz_per_s_per_W": res.grooving.alpha,
            "gamma_Hz_per_s_per_K": res.grooving.gamma,
            "se_alpha": res.grooving.se_alpha,
            "r_squared": res.grooving.r_squared,
            "power_kind": res.grooving.power_kind,
        },
        "alpha_bound_95_Hz_per_s_per_W": res.alpha_bound_95,
        "dn_per_year_bound_95": res.dn_per_year_bound_95,
        "notes": res.notes,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
