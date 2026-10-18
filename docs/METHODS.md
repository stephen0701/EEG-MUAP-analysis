# Methods and implementation notes

[README](../README.md) · [Methods](METHODS.md) · [Results](RESULTS.md) · [Reproduction](REPRODUCIBILITY.md) · [Project notes](PROJECT_NOTES.md)

## EMG deconvolution

`process_single_dataset` in `EMG_to_MUAP.py` reads the first `EMG/*_align.csv`, maps LGL/RGL to LGA/RGA, and processes requested muscle columns independently. Eight input channels do not make this HD-EMG decomposition: each channel is deconvolved separately.

`AlternativeProcessor` removes the mean and trend, applies a 60 Hz notch, and uses 40 Hz high-pass and 450 Hz low-pass filtering at 1000 Hz. PSD fitting estimates derivative-Gaussian kernel widths. `single` uses three PSD-weighted percentile locations; `multi` uses two equal-width, opposite-polarity kernels, spanning approximately ±25 ms. Degenerate/short-signal fallback branches return a different shape from the normal `(sigmas, diagnostics)` result and need investigation.

The solver applies IRLS to an augmented residual using convolution operators and SciPy `lsmr`. Regularization is `alpha = 0.01 * lambda_max(A.T A)`. The high-level route uses at most 10 IRLS iterations, relative tolerance `1e-6`, and a nonnegative projection after each solve. These calculations were preserved.

CWF sums firing-pattern coefficients weighted by kernel RMS; `CWF_comp` stores per-component weighted signals. The author confirmed that device EMG input is in μV and kernel amplitudes received no additional calibration. Kernel scale is part of the mathematical model and must not be interpreted as a calibrated individual MUAP voltage, despite the μV label on original kernel plots. Recovered components are not confirmed individual motor units.

## Candidate spikes and alignment

The reported experiment used a 97th-percentile amplitude threshold and 30 ms minimum ISI. `process_single_dataset_spikes` uses those defaults. The lower-level converter also supports manual/MAD thresholds, smoothing, prominence, and amplitude-based ISI suppression. Record those settings rather than assuming they are interchangeable.

The executable example excludes `[0, 8)` seconds. The research summary does not establish that exclusion for every result. Rate statistics use the interval between first and last detected spikes, not the full recording.

The main motion detector creates aligned EEG/EMG files and movement results. The neural segmentor interpolates CWF components from 1000 to 500 Hz and uses relative time per phase. Spike trains retain their original sampling rate; spike times are in seconds. Actual synchronization accuracy must be verified from recordings. Four supported phases are STS, STW, steady gait, and WTS.

## Coherence

`CoherenceAnalyzer.compute_coherence` calls SciPy `signal.coherence` with `nperseg=256`, `noverlap=128`, `nfft=512`, then takes its square root and retains frequencies up to 50 Hz. Higher-level analysis skips pairs shorter than 512 samples. Band means include both endpoints.

This is coherence magnitude, not magnitude-squared coherence or a percentage correlation. No significance threshold is computed. Table 6.1 means are retained as reported in the project research summary; exact aggregation and original source revision cannot be independently established.

| Band | Current coherence/STA/PPC defaults (Hz) |
| --- | --- |
| Delta | 1-4 |
| Theta | 4-8 |
| Alpha / figure label Mu | 8-12 |
| Beta | 12-30 |
| Low Gamma | 30-46 |

These differ from the synergy repository's EEG topography bands and must not be silently unified.

## STA and PPC

STA bandpass-filters EEG, extracts windows, and calculates mean, SD, and SEM. Half-windows are 0.5, 0.5, 0.3, 0.15, and 0.075 seconds across the bands above. Boundary spikes without a full window are omitted from STA; PPC independently samples Hilbert phase at spike times.

The raw PPC expression is `(abs(sum(exp(1j * phase)))**2 - N) / (N * (N - 1))`. The implementation returns `clip(ppc, 0, 1)`. Negative finite-sample values therefore appear as zero. These outputs should not be described as an unchanged unbiased estimator, and zero cells do not prove absent coupling.

**STA display limitation:** extraction returns a physical `relative_time` vector and a separate `display_time` spanning ±0.3 regardless of window. The author confirmed the original figures used this common display axis. They are included in the [results gallery](RESULTS.md) with an explicit warning that the displayed tick labels are not a common physical-lag scale across bands. They cannot support comparisons of actual peak latency or oscillation frequency from that x-axis. Analysis, image pixels, and time axes were not silently regenerated or relabeled.

## Evidence boundary

The research summary supports reported scope and summary findings; image inspection supports descriptive labels and isolated PPC values. Recordings, ground-truth motor-unit identities, complete parameter logs, and the underlying Table 6.1 data are unavailable. Software verification does not establish scientific or clinical validation.
