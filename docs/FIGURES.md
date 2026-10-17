# Figure provenance and selection

Source: owner-supplied `Research_Repo_Codex_Pack.zip`, reference figures from the capstone chapters summarized in the supplied brief. The owner authorized inclusion in these private repositories on 2026-10-17. Public-release permission was not established.

Selected originals are copied byte-for-byte. Filenames are descriptive; no pixels, axes, or underlying values were edited. No source measurements were available for regeneration. Visual checking validates visible text, not the full analysis provenance.

| Included file | Original | Dimensions | SHA-256 |
| --- | --- | --- | --- |
| `spike_detection.jpeg` | `fig_6_2.jpeg` | 747 × 576 | `a98a39465baca39e0241b9bb2ea0f751de088e62b287c762ca9b855fad506c59` |
| `ppc_single_model.jpeg` | `fig_6_4a.jpeg` | 1350 × 389 | `c85303a5142897cca7f0186cc1030aba13922fe06692623a93bc2283c3308adc` |
| `ppc_multi_model.jpeg` | `fig_6_4b.jpeg` | 1350 × 573 | `396c2dc6da6ffe09b1fcea4cfac93a064f48b2541ae6442771f6bd2e1952e777` |

## Label checks and exclusions

- Fig. 6.2: visible LBF, component, time, amplitude, raster, and ISI labels were inspected. The detected-spike span determines the displayed rate. Neither spike identity nor the exact run configuration can be recovered from the image alone.
- Fig. 6.4a/b: STS, single/multi model, Cz, EMG components, spike counts, and bands agree with source defaults. “Mu” labels the 8-12 Hz band named `alpha` internally. The scale matches clipped PPC, not a claim of statistical significance. `n` counts spikes, not participants.
- Fig. 6.1a/b: excluded because kernels are not independently amplitude-calibrated, while the image axes show microvolts. Model structure is explained in text instead.
- Fig. 6.3a/b: excluded because the source maps frequency-dependent extraction windows onto a fixed ±0.3 s display axis. Correct physical-lag plots require verified regeneration, not pixel relabeling.

Checksums identify supplied image bytes; they do not prove that the figures came from a specific code commit. See [Methods](METHODS.md) for interpretation and [Data/publication](DATA_AND_PUBLICATION.md) for release limits.
