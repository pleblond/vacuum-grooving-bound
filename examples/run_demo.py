"""End-to-end demo: synthetic log -> Protocol v3.1 bound (null + injected)."""

from vacuum_creep.pipeline import PipelineConfig, run_pipeline
from vacuum_creep.synthetic import make_synthetic


def main() -> None:
    for label, alpha in (("null (alpha=0)", 0.0), ("injected (alpha=3e-4)", 3e-4)):
        df = make_synthetic(n_epochs=10, alpha=alpha, seed=3)
        res = run_pipeline(df, PipelineConfig())
        g = res.grooving
        print(f"\n== {label} ==")
        print(f"epochs: {len(res.epochs)}")
        if g is not None:
            print(f"alpha_hat = {g.alpha:.3g} +- {g.se_alpha:.3g} Hz/s/W ({g.power_kind})")
            print(f"gamma_hat = {g.gamma:.3g} +- {g.se_gamma:.3g} Hz/s/K, R2={g.r_squared:.3f}")
        print(f"95% bound: |alpha| <= {res.alpha_bound_95:.3g} Hz/s/W")
        print(f"95% bound: dn/yr <= {res.dn_per_year_bound_95:.3g} per 100 uW")
        for note in res.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
