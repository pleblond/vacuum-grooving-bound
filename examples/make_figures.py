"""Generate publication figures for the grooving note (PDF, vector).

Fig. 1: redigitized Lee et al. 2026 Fig. 3b (drift magnitude vs cavity age)
        with the paper's stated recent rates overlaid as stars.
Fig. 2: Si2 aging treatments compared: windowed medians (split day 2000)
        vs joint exponential and 1/t fits over days >= 800.

Outputs: docs/figures/fig1_digitized.pdf, docs/figures/fig2_si2_aging.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vacuum_creep import literature as L

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "docs" / "figures"
FIGDIR.mkdir(exist_ok=True)

COLORS = {"Si2": "black", "Si3": "red", "Si5": "blue", "Si6": "purple"}
SIGNS = {"Si2": "−", "Si3": "−", "Si5": "−", "Si6": "+"}


def fig1(df) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for cav, g in df.groupby("cavity"):
        ax.plot(
            g["days_since_contacting"],
            g["drift_rate_uHz_s"].abs(),
            ".",
            ms=3,
            color=COLORS[cav],
            label=f"{cav} ({SIGNS[cav]})",
        )
        a = L.PAPER_ANCHORS[cav]
        ax.plot(
            g["days_since_contacting"].max(),
            abs(a["recent_uHz_s"]),
            marker="*",
            ms=9,
            color=COLORS[cav],
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("days since optical contacting")
    ax.set_ylabel(r"$|$drift rate$|$ ($\mu$Hz/s)")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_digitized.pdf")
    print("wrote", FIGDIR / "fig1_digitized.pdf")


def fig2(df) -> None:
    sub = df.loc[df["cavity"] == "Si2"].sort_values("days_since_contacting")
    t = sub["days_since_contacting"].to_numpy(float)
    r = sub["drift_rate_uHz_s"].to_numpy(float)
    b0e, b1e, taue, _, _ = L.aging_fit_exp(df, "Si2", 800.0)
    b0t, b1t, _, _ = L.aging_fit_1overT(df, "Si2", 800.0)
    w = L.windowed_stats(df, "Si2", 2000.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(t, -r, ".", ms=4, color="black", label="Si2 digitized")
    tg = np.logspace(np.log10(800), np.log10(t.max()), 200)
    ax.plot(tg, -(b0e + b1e * np.exp(-tg / taue)), "-", color="red",
            label=f"exp (tau={taue:.0f} d)")
    ax.plot(tg, -(b0t + b1t / tg), ":", color="gray", label="1/t")
    for key, span, ls in (("early", (t.min(), 2000.0), "--"), ("late", (2000.0, t.max()), "--")):
        med = w[key][0]
        ax.hlines(-med, span[0], span[1], colors="blue", linestyles=ls)
    ax.axvline(2000.0, color="blue", lw=0.8, alpha=0.6)
    ax.text(2150, 160, "window medians", color="blue", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("days since optical contacting")
    ax.set_ylabel(r"$-$drift rate ($\mu$Hz/s)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_si2_aging.pdf")
    print("wrote", FIGDIR / "fig2_si2_aging.pdf")


def main() -> None:
    df = L.load_digitized()
    if df is None:
        raise SystemExit("digitized CSV missing; run scripts/digitize_fig3b.py first")
    fig1(df)
    fig2(df)


if __name__ == "__main__":
    main()
