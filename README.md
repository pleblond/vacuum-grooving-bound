# Bounding DC Vacuum Index Creep — Protocol v3.1 Implementation

Runnable implementation of **Protocol v3.1** (see
`docs/LIGO_Vacuum_Creep_Protocol_v3_1_Cavity_Drift.tex`):
bounding slow vacuum-index drift ("grooving") from cryogenic silicon
cavity AOM-correction logs.

## Idea in one paragraph

Clock ratios (Yb+/Sr) servo away cavity drift, so they cannot see it. The
servo's **AOM correction** `ν_corr(t)` is atoms-minus-cavity: a raw DC ruler
for cavity length + vacuum index. Regressing per-epoch drift rates against
circulating power bounds the power-dependent vacuum effect:

```text
s_i = β_Si + α·⟨P⟩_i + γ·ΔT_i + ε_i        (Eq 1)
```

`α = 0` under the null; a 95% upper bound converts to `Δn/yr per 100 µW`.

## The three v3.1 corrections (all implemented)

| # | Issue in v3.0 | Fix here |
|---|---|---|
| R1 thermal | residual T coupling omitted | `γ·ΔT_i` kept as regressor even at 4 K / 124 K zero-crossing; never pre-subtracted |
| R2 unwrap | `np.unwrap` applied to Hz data | slopes fit on **raw frequency offsets**; unlock steps handled by epoch grouping |
| R3 finesse | assumed constant `F` | interpolated ring-down `F_measured` when logged, else nominal `F` (documented ~2% error) or direct `P_trans` proxy mode |

## Layout

```text
src/vacuum_creep/
  pipeline.py    Phases 1–4: ingest → epoch slopes → WLS fit → bound
  synthetic.py   synthetic log generator with known injected α
  cli.py         command-line runner (CSV or --synthetic)
docs/            the protocol source (.tex)
examples/run_demo.py   null + injected end-to-end demo
tests/           30 pytest checks (recovery, deconfounding, validation)
```

## Quick start

```bash
pip install -e .
python examples/run_demo.py
python -m vacuum_creep.cli --synthetic
python -m vacuum_creep.cli --input your_log.csv --output bound.json
python -m pytest tests/ -q
```

Input CSV columns: `time_s, nu_corr_Hz, P_trans_W, temp_K, is_on`
(optional `F_measured`). Flags: `--trans-proxy` (R3 fallback regressing on
`P_trans` directly), `--min-epoch-s`, `--output` (JSON).

## Validation status

- 29/30 pytest checks pass: null gives a tight bound, injected
  `α = 3e-4 Hz/s/W` is recovered to ~20%, `γ` recovers its true value,
  finesse/proxy modes agree in sign, short epochs fail safe (no fit, no crash).
- Demo output (10 synthetic epochs, seed 3): null → `|α| ≤ 1.5e-04`,
  injected → `α̂ = 2.4e-04 ± 4.3e-05`, `R² > 0.999`.

## Scale note

Demo bounds (`Δn/yr ~ 1e-14`) come from ~10 short synthetic epochs. The
protocol's `1e-19/yr` target assumes a multi-year open dataset at
`1e-13 torr`, 4 K, where `β_Si → 0` breaks the `β`/`α` degeneracy and long
integration beats down `SE(α)`.

## First literature analysis: Lee et al. 2026 Fig. 3b

`examples/Lee2026_Fig3b.ipynb` digitizes the published 10-year drift records of
four cryogenic Si cavities (Si2/Si3/Si5 at 124 K, Si6 at 17 K):

- `scripts/digitize_fig3b.py` fetches arXiv:2509.13503v1 and digitizes Fig. 3b
  into `data/lee2026_fig3b.csv` (the source ships figures only — no CSV supplement).
- Endpoints reproduce the paper: Si2 −52, Si3 −15, Si6 +10 µHz/s;
  Si2 integrates to −29.5 kHz over days 260–3798 (paper −44 kHz/10 yr;
  the gap implies −646 µHz/s mean over the steep un-digitized days 0–260,
  consistent with Si3's −883 at day 269).
- Aging + reach utilities: `windowed_stats` (option (a)), `aging_fit_exp`
  (option (b); Si2 τ≈1.8 yr, beats 1/t), `dither_sensitivity` and
  `slope_precision_allan` (first-principles dither reach).
- `src/vacuum_creep/literature.py` holds the paper anchors, unit conversions,
  and the constant-power sensitivity projection.
- Aging + thermal options: `beta_window_edges` fits a separate β per time
  window (paper option (a)); `fit_quadratic_thermal` adds γ₂ΔT² for the
  124 K crossing; `unwrap_beat_phase` is provided for raw beat-phase logs only.
- Honest limit: at constant power α is degenerate with β_Si, so the notebook
  shows what logged power excursions would buy, plus a ready-to-send
  P_trans(t) + T(t) data request for PTB/JILA.

## To run on real lab data

1. Export `ν_corr` (subtract nominal AOM, e.g. 80 MHz, or pass it through —
   a constant offset does not affect slopes), `P_trans`, cryostat `T`,
   lock flag, and ring-down finesse if available.
2. Run the CLI; check `R²`, `γ` consistency with zero at 4 K, and the
   cumulative `E_cum` cross-check line in the notes.
3. Quote `dn_per_year_bound_95` per 100 µW with the finesse caveat from R3.

## Cite this work

If you use this code or analysis, please cite the software and the draft
(see `CITATION.cff`, rendered by GitHub as "Cite this repository"):

```bibtex
@software{Leblond_vacuum_grooving_bound_2026,
  author  = {Leblond, Philippe},
  title   = {{vacuum-grooving-bound}},
  url     = {https://github.com/pleblond/vacuum-grooving-bound},
  license = {MIT},
  year    = {2026}
}

@unpublished{Leblond_grooving_note_2026,
  author = {Leblond, Philippe},
  title  = {{First search for cumulative vacuum index creep using cryogenic
             silicon cavity drift logs: open-data analysis and sensitivity
             projection}},
  year   = {2026},
  note   = {Draft, \url{https://github.com/pleblond/vacuum-grooving-bound}}
}
```

Analyses built on the digitized drift record should also cite Lee et al.,
PRL 136, 033801 (2026), arXiv:2509.13503.
