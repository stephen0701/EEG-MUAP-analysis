# Figure provenance and interpretation

Source: owner-supplied `Research_Repo_Codex_Pack.zip`, reference figures associated with the capstone chapters summarized in the supplied brief. The owner authorized all supplied figures for these private portfolio repositories, including originals with explicit corrections or interpretation limits. Public-release permission was not established.

All original image bytes are preserved. Descriptive filenames do not alter the plotted data. Labels requiring correction are addressed immediately beside the figures; no pixels, axes, coefficients, or curves were edited or regenerated. Full code-to-figure provenance cannot be independently reconstructed without the original analysis products.

| Source figure | Repository asset | Dimensions |
| --- | --- | --- |
| fig_6_1a | [kernel_single_model.jpeg](figures/kernel_single_model.jpeg) | 617 × 713 |
| fig_6_1b | [kernel_multi_model.jpeg](figures/kernel_multi_model.jpeg) | 851 × 643 |
| fig_6_2 | [spike_detection.jpeg](figures/spike_detection.jpeg) | 747 × 576 |
| fig_6_3a | [sta_single_model.jpeg](figures/sta_single_model.jpeg) | 1685 × 781 |
| fig_6_3b | [sta_multi_model.jpeg](figures/sta_multi_model.jpeg) | 1688 × 939 |
| fig_6_4a | [ppc_single_model.jpeg](figures/ppc_single_model.jpeg) | 1350 × 389 |
| fig_6_4b | [ppc_multi_model.jpeg](figures/ppc_multi_model.jpeg) | 1350 × 573 |

SHA-256, metadata inventory, and source-to-file mappings are stored in [manifest.json](figures/manifest.json). No EXIF tags were found in the supplied JPEGs. Hashes verify byte identity, not the scientific validity of the results.

## Corrections and owner clarifications

- **Figs. 6.1a/b:** the owner confirmed input EMG is in μV and the generated kernels received no extra amplitude calibration. Original kernel axes retain μV labels, but kernel height is treated as a mathematical model scale, not a calibrated individual MUAP voltage. Component count does not identify distinct physiological motor units.
- **Fig. 6.2:** LBF, component, threshold, raster, and ISI labels were inspected. The displayed rate is calculated over the detected-spike span. The image does not recover every run setting or validate motor-unit identities.
- **Figs. 6.3a/b:** the owner confirmed frequency-specific windows were placed on a common ±0.3 display axis. Those ticks cannot be treated as a common physical-time scale across bands or used for actual latency/frequency comparisons. Original images remain intact, with the restriction prominently stated in the gallery.
- **Figs. 6.4a/b:** STS, single/multi model, Cz, muscle/component labels, counts, and bands agree with source defaults. Mu denotes the 8-12 Hz band named alpha internally. Values follow the clipped PPC convention; `n` counts sampled spikes, not participants. Isolated larger cells are not significance claims.

The README emphasizes both kernel models, spike retrieval, and a representative PPC figure. The [complete results gallery](RESULTS.md) retains both STA and PPC models alongside the reported coherence table.

Owner clarifications are documented statements rather than independent source-data verification. See [Methods](METHODS.md), [Verification](VERIFICATION.md), and [Data/publication](DATA_AND_PUBLICATION.md) for scope and limits.
