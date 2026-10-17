# Verification record

Prepared 2026-10-17 against main commit `fa3cdf80bb3241b01c5778b77c93d498a71f3ffa`.

The Python/MATLAB sources were reviewed to identify analysis functions, data contracts, configurable thresholds, and exploratory alternatives. Python checks use synthetic signals only; none validate physiological motor-unit recovery or reproduce report findings.

## Completed Python checks

- All nine Python modules parsed and imported successfully on Python 3.11.8.
- AST comparison against the baseline source found 41 top-level analysis functions/classes unchanged. Example `main`/path configuration is excluded; numerical class methods are included.
- Derivative-Gaussian kernels have the expected 51-sample shape and opposite-polarity relation in `multi` mode.
- Synthetic competing peaks confirm amplitude-based suppression and the 30 ms minimum ISI.
- Seeded noise confirms coherence output equals the square root of SciPy's MSC.
- Synthetic phases confirm clipped PPC behavior. STA extraction confirms the documented physical/display-axis mismatch.
- A 512-sample synthetic deconvolution with one IRLS iteration produced two finite components; its built-in adjoint checks passed 5/5 trials. This is a software smoke check, not recovery validation.
- Relative Markdown links, machine-path removal, image hashes, and diff whitespace were checked.

Check environment: NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6, Matplotlib 3.11.2, seaborn 0.13.2. Direct dependencies are pinned in `requirements.txt` to this check environment, not claimed as the original experiment versions. [Machine-readable software checks](software_checks.json) preserve the results.

MATLAB R2024a was installed, but initial startup failed with `failed to load settings errors_warnings plugin`; retrying with a workspace preference directory failed to communicate with required MathWorks services (5201). MATLAB static diagnostics and EEG preprocessing execution were **not completed**. EEGLAB/ASR, manual ICA decisions, and the original montage were not supplied as a complete runnable experiment.

All seven supplied Chapter 6 images were visually checked and copied unchanged, not regenerated. The owner confirmed device EMG units of μV, no additional kernel amplitude calibration, and use of a common STA display axis. Kernel figures are included with the input/model-scale distinction; STA figures are included in [Results](RESULTS.md) with explicit physical-lag restrictions. Coherence means were transcribed from the supplied brief; original Table 6.1 data and aggregation logs remain unavailable.

The figure-restoration revision changes documentation and image assets only. Analysis source files and parameters are unchanged from the preceding portfolio revision. The Python checks above remain the previous software-check record; no participant analysis was newly run.

No participant recording or sensitive metadata was used for software checks. See [Methods](METHODS.md) for numerical conventions and unresolved edge cases.
