# First search for cumulative vacuum index creep

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22886215.svg)](https://doi.org/10.5281/zenodo.22886215)

Companion code for **"First search for cumulative vacuum index creep
('vacuum grooving') in cryogenic silicon cavity drift records"**
([paper PDF](docs/vacuum_grooving_note.pdf), [source](docs/vacuum_grooving_note.tex)):
bounding slow vacuum-index drift ("grooving") from cryogenic silicon cavity
AOM-correction logs.

Existing vacuum-nonlinearity searches test only instantaneous responses,
assuming a vacuum with zero memory; cavity drift is treated as background
and subtracted. This search is different: it looks for persistence ---
whether integrated photon dose leaves a lasting imprint on the vacuum
index. No prior study has tested or bounded it; this repository establishes
the first quantitative upper bounds from published open drift records.

## Background

Clock ratios (Yb+/Sr) servo away cavity drift, so they cannot see it. The
servo's **AOM correction** `ν_corr(t)` is atoms-minus-cavity: a raw DC ruler
for cavity length plus vacuum index. Regressing per-epoch drift rates
against circulating power bounds any power-dependent vacuum effect:

```text
s_i = β_Si(t) + α·⟨P⟩_i + γ₁·ΔT_i + γ₂·ΔT_i² + ε_i
```

`α = 0` under the null; its 95% upper bound converts to `Δn/yr per 100 µW`.
Temperature enters as regressors (never pre-subtracted), slopes are fit on
raw frequency offsets within locked epochs, and circulating power uses
interpolated ring-down finesse when logged, else transmitted power as proxy.
Silicon aging is handled by windowed intercepts or a joint exponential fit.

## Layout

```text
src/vacuum_creep/
  pipeline.py    ingest → epoch slopes → WLS fit → bound (+ aging options)
  synthetic.py   synthetic log generator with known injected coefficients
  literature.py  Lee et al. 2026 anchors, conversions, aging and dither tools
  cli.py         command-line runner (CSV or --synthetic)
scripts/
  digitize_fig3b.py   fetch arXiv:2509.13503v1, digitize Fig. 3b → data/
data/
  lee2026_fig3b.csv        digitized drift histories (Si2/Si3/Si5/Si6)
  lee2026_fig3b_meta.json  digitization provenance
docs/
  vacuum_grooving_note.{tex,pdf}   the paper
  figures/                         paper figures (vector PDF)
examples/
  Lee2026_Fig3b.ipynb   end-to-end analysis notebook
  make_figures.py       regenerates the paper figures
  run_demo.py           synthetic null + injected demo
tests/                  30 pytest checks
```

## Quick start

```bash
pip install -e .
python examples/run_demo.py                    # synthetic demo
python -m vacuum_creep.cli --synthetic         # CLI on synthetic data
python -m vacuum_creep.cli --input log.csv --output bound.json
python -m pytest tests/ -q                     # test suite
python examples/make_figures.py                # rebuild paper figures
```

Input CSV columns: `time_s, nu_corr_Hz, P_trans_W, temp_K, is_on`
(optional `F_measured`). Flags: `--trans-proxy` (regress on `P_trans`
directly), `--min-epoch-s`, `--output` (JSON).

## Status

- Synthetic validation: null consistent with zero; injected
  `α = 3e-4` recovered (2.36e-04 single-fit; 2.99e-04 windowed, 3.04e-04
  joint-exponential where single-β flips sign); thermal and finesse
  handling verified; short epochs fail safe.
- Literature: digitized endpoints reproduce Lee et al. (Si2 −52, Si3 −15,
  Si6 +10 µHz/s); Si2 integrates to −29.5 kHz over days 260–3798
  (published −44 kHz/10 yr, reconciled via the steep early era).
- Lead conditional bound: `Δn/yr < 1.5e-13` per 100 µW (Si6, ≥10%
  excursion assumption). See the paper for the full ladder, systematics,
  and dither projections.

## To run on real lab data

1. Export `ν_corr` (subtract nominal AOM, e.g. 80 MHz, or pass it through —
   a constant offset does not affect slopes), `P_trans`, cryostat `T`,
   lock flag, and ring-down finesse if available, ideally at 10-day means.
2. Run the CLI; check `R²`, the thermal coefficients, and the cumulative
   `E_cum` cross-check line in the notes.
3. Quote `dn_per_year_bound_95` per 100 µW with the finesse caveat
   (nominal `F` is approximate; ring-down `F(t)` preferred).

## Cite this work

If you use this code or analysis, please cite the software and the paper
(see `CITATION.cff`, rendered by GitHub as "Cite this repository"):

```bibtex
@software{Leblond_vacuum_grooving_bound_2026,
  author  = {Leblond, Philippe},
  title   = {{vacuum-grooving-bound}},
  url     = {https://github.com/pleblond/vacuum-grooving-bound},
  doi     = {10.5281/zenodo.22886215},
  license = {MIT},
  year    = {2026}
}

@unpublished{Leblond_grooving_note_2026,
  author = {Leblond, Philippe},
  title  = {{First search for cumulative vacuum index creep ('vacuum
             grooving') in cryogenic silicon cavity drift records}},
  year   = {2026},
  note   = {Draft, \url{https://github.com/pleblond/vacuum-grooving-bound}}
}
```

Analyses built on the digitized drift record should also cite Lee et al.,
PRL 136, 033801 (2026), arXiv:2509.13503.

## License

Code and analysis scripts: MIT (see `LICENSE`). Paper draft
(`docs/vacuum_grooving_note.*`, `docs/figures/`): CC-BY-4.0
(https://creativecommons.org/licenses/by/4.0/) — reuse with attribution.
Digitized values are facts extracted from Lee et al. 2026 — please cite
that paper alongside this repository.
