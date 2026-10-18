# Project notes

[README](../README.md) · [Methods](METHODS.md) · [Results](RESULTS.md) · [Reproduction](REPRODUCIBILITY.md) · [Figure provenance](FIGURES.md)

## Attribution and sources

**Sing-Hao Wang** primarily implemented the code, integrated the workflow, analyzed the results, and prepared the research presentation. **Dr. Tsai** contributed clinical interpretation. These roles were confirmed by the project owner; they do not assign module-level authorship or credit upstream algorithms to this project.

**Research source:** Sing-Hao Wang, *Toward Digital Twin-Centered Human-Exoskeleton Collaboration: From Multi-Resolution Corticomuscular Coupling Analyses to an Incremental Mixture-of-Experts Locomotion Controller*, undergraduate capstone report, June 2026, Chapter 6, pp. 53–61. Scope and Table 6.1 values were transcribed from the supplied `RESEARCH_BRIEF.md`; the full report, source result table, and aggregation logs were not independently verified.

External methods and software:

- [SciPy coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html) returns magnitude-squared coherence; this source takes its square root.
- Vinck et al., [*The pairwise phase consistency: a bias-free measure of rhythmic neuronal synchronization*](https://doi.org/10.1016/j.neuroimage.2010.01.073), *NeuroImage* 51(1), 112–122 (2010). This implementation additionally clips finite-sample PPC to `[0, 1]`; the documentation preserves that distinction.
- [EEGLAB](https://eeglab.org/) and [clean_rawdata / ASR](https://eeglab.org/plugins/clean_rawdata/) provide EEG processing, cleaning, and manual ICA infrastructure. Original experiment revisions are unknown.
- NumPy, SciPy (`LinearOperator`, `lsmr`, filtering, Hilbert transform), pandas, Matplotlib, and seaborn provide numerical and visualization infrastructure.

The source's single-motor-unit corticomuscular-coherence reference lacks a complete bibliographic record. The exact citation and original multi-kernel method's provenance remain to be recovered. gCKC is future work, not implemented functionality. No external toolbox source or new license was added.

## Data and release status

The owner confirmed on **October 18, 2026** that both research repositories are approved for public release. **Repository visibility remains unchanged pending the owner's review of this cleanup.** This confirmation supersedes the earlier private-portfolio-only wording; it does not initiate a visibility change.

All seven authorized Chapter 6 figures remain byte-for-byte unchanged in `figures/`, with the original kernel-unit and STA-axis caveats retained beside the figures. The earlier undocumented externally hosted attachment is not used as evidence. No participant recordings, identifiers, or new metadata tables were added, and the supplied brief remains outside the repository.

Keep authorized data and generated products outside the checkout. `.gitignore` covers common recording formats, pickles, NumPy arrays, motion files, spreadsheets, videos, local settings, and result directories. It does not remove tracked content or protect Git history automatically. Load only trusted local pickle files. No history rewrite was performed.

Public-release approval does not select a software or figure license. No new license is inferred or assigned, and external software retains its own terms.

## Verification record

### Final cleanup — October 18, 2026

Baseline: main commit `da67e60e1c6751fbb1b03a9b44b66233e3d6add0`.

- Source edits are limited to English comments. The complete syntax trees of all nine Python modules are identical to the baseline, including example entry points, strings, and parameters.
- Across both projects, all eight MATLAB files retain identical text after removing comments; original executable statements and disabled example code are preserved.
- Figure checksums and requirements are unchanged. Documentation links and all seven gallery images are checked against the final repository tree.

These source-preservation checks do not establish physiological validity or reproduce report findings.

### Earlier software checks — October 17, 2026

The [machine-readable record](software_checks.json) retains the earlier checks against commit `fa3cdf80bb3241b01c5778b77c93d498a71f3ffa`:

- Nine Python modules parsed/imported, with 41 top-level analysis functions/classes unchanged; example path configuration was excluded from that earlier comparison.
- Multi-mode derivative-Gaussian kernels had 51 samples, opposite polarity, and zero-sum shape.
- Synthetic competing peaks exercised amplitude-based suppression and the 30 ms minimum ISI.
- Seeded synthetic noise confirmed coherence equals `sqrt(SciPy MSC)`; synthetic phases confirmed clipped PPC and the physical/display-time distinction in STA.
- One IRLS iteration on 512 synthetic samples produced two finite components; built-in adjoint checks passed 5/5 trials.

Those synthetic checks were not rerun for this comment-only source cleanup. Their environment was Python 3.11.8, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6, Matplotlib 3.11.2, and seaborn 0.13.2. Requirements describe this check environment, not recovered experiment versions.

MATLAB R2024a startup and MathWorks-service errors prevented static diagnostics and EEG preprocessing execution. The original montage, complete EEGLAB/ASR environment, and manual ICA record were not supplied as a runnable experiment.

All seven images were visually checked in the earlier review. The owner confirmed input EMG units, absence of extra kernel calibration, and the common STA display axis. Those clarifications and the reported coherence means remain distinct from independent data verification. See [Methods](METHODS.md) for conventions and unresolved edge cases.
