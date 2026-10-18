# Exploratory EEG–MUAP Analysis

An evaluation of EMG-derived firing patterns and their relationship to synchronized EEG.

[Results and figures](docs/RESULTS.md) · [Methods](docs/METHODS.md) · [Usage](docs/REPRODUCIBILITY.md) · [Project notes](docs/PROJECT_NOTES.md)

## Research objective and scope

This undergraduate project tests whether single-channel EMG deconvolution produces motor-unit-like firing patterns with interpretable cortical relationships. It compares two kernel models using EEG–cumulative weighted firing (CWF) coherence, spike-triggered averages (STA), and pairwise phase consistency (PPC).

The study examines **one representative participant/recording**. Findings were **inconclusive**: no validated individual motor-unit spike trains, MU-specific neural decoder, or consistent EEG–MUAP phase locking was established. The wider 15-participant acquisition cohort does not make these group-level results.

## Contributions

**Sing-Hao Wang** primarily implemented the code, integrated the workflow, analyzed the results, and prepared the research presentation. **Dr. Tsai** contributed clinical interpretation. The contribution is implementation and critical evaluation of an exploratory approach; external algorithms are credited separately in [project notes](docs/PROJECT_NOTES.md#attribution-and-sources).

## Methodology

1. Preprocess and align **eight-channel sEMG (1000 Hz)**, **32-channel EEG (500 Hz)**, and **motion capture (30 Hz)**.
2. Deconvolve each EMG channel separately with derivative-Gaussian kernels: three kernels for the `single` model, or two opposite-polarity kernels for `multi`.
3. Construct CWF and candidate spikes; the reported spike experiment uses a 97th-percentile threshold and 30 ms minimum ISI.
4. Segment aligned movement phases and calculate coherence magnitude, STA, and PPC against EEG.

The [methods guide](docs/METHODS.md) documents solver parameters and preserved conventions: coherence is `sqrt(MSC)`, PPC is clipped to `[0, 1]`, and STA uses frequency-specific extraction windows mapped to a shared display axis.

## Key results

The supplied brief reports mean EEG–CWF coherence values of **0.352–0.370** across five EEG bands, offering weak band-specific discrimination. STA showed modest fluctuations without pronounced band-specific patterns. PPC was near zero for most combinations, with isolated values around **0.350–0.377** rather than consistent phase locking.

These observations were not independently recomputed from participant data. The [results gallery](docs/RESULTS.md) retains the exact five-band coherence table, all seven figures, and their interpretation limits.

## Representative figures

![Three-kernel model and recovered firing-pattern coefficients for LBF](docs/figures/kernel_single_model.jpeg)

**Three-kernel model (Fig. 6.1a), LBF: left biceps femoris.** Device EMG is in μV, but the generated kernels received no additional amplitude calibration. Their retained μV axes must not be read as calibrated individual motor-unit voltages; components are model outputs.

![STS pairwise phase consistency for the three-kernel model at EEG electrode Cz](docs/figures/ppc_single_model.jpeg)

**PPC at Cz during STS (Fig. 6.4a).** `n` counts sampled spike phases, not participants. Values use the clipped PPC convention; isolated higher cells do not establish statistical significance.

Both kernel models, spike diagnostics, both STA models, and both PPC models remain in the [complete gallery](docs/RESULTS.md). The STA figures' common ±0.3 display axis **cannot be used to compare actual peak latencies across bands**.

## Usage and repository guide

Python dependencies are listed in [`requirements.txt`](requirements.txt). From the repository root:

```console
python -m venv .venv
```

Activate the environment, then run `python -m pip install -r requirements.txt`. Follow the [reproduction guide](docs/REPRODUCIBILITY.md) for input contracts, environment variables, and function-level examples. The repository contains research modules, not a packaged CLI or end-to-end driver. MATLAB EEG preprocessing also requires compatible toolboxes, EEGLAB/ASR, the acquisition montage, and manual ICA review.

| Location | Purpose |
| --- | --- |
| [EEG_preprocessing/](EEG_preprocessing/) | MATLAB EEG class and manual ICA example. |
| [tool_kits/](tool_kits/) | EMG processing, movement alignment, deconvolution, spikes, phase segmentation, coherence, and STA/PPC. |
| [docs/](docs/) | Methods, full results, reproduction, figure provenance, and project notes. |

The reproduction guide identifies the main route and the `movement_state_detector_v2.py` / `spike_triggered_analysis_new.py` alternatives. Existing filenames and imports are retained.

## Limitations

Single-channel surface EMG does not resolve overlapping motor units reliably. Kernel identity, spike thresholds, alignment, case-level sampling, clipped PPC, and the STA display axis limit interpretation. Recordings and ground-truth motor-unit identities are unavailable. HD-EMG, validated decomposition, sensitivity analysis, and broader validation remain future work.

[Project notes](docs/PROJECT_NOTES.md) consolidate attribution, completed software checks, and data/release status. Related project: [EEG-muscle-synergies-analysis](https://github.com/stephen0701/EEG-muscle-synergies-analysis).
