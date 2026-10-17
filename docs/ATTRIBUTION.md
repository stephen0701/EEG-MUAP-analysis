# Attribution and references

**Sing-Hao Wang:** primary code implementation, workflow integration, analysis, and research presentation. **Dr. Tsai:** clinical interpretation of the MUAP study. These roles were confirmed by the project owner during portfolio preparation. Do not infer a specific module's authorship beyond that confirmation or attribute upstream algorithms to this project.

**Research source:** Sing-Hao Wang, *Toward Digital Twin-Centered Human-Exoskeleton Collaboration: From Multi-Resolution Corticomuscular Coupling Analyses to an Incremental Mixture-of-Experts Locomotion Controller*, undergraduate capstone report, June 2026, Chapter 6, pp. 53-61. Scope and reported Table 6.1 values were transcribed from the supplied `RESEARCH_BRIEF.md`; the full report and underlying result table were not independently verified.

**External methods and software:**

- [SciPy coherence documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html): the library returns magnitude-squared coherence. This repository takes its square root.
- Vinck et al., [*The pairwise phase consistency: a bias-free measure of rhythmic neuronal synchronization*](https://doi.org/10.1016/j.neuroimage.2010.01.073), NeuroImage 51(1), 112-122 (2010). The repository additionally clips finite-sample PPC to `[0, 1]`; it is not an unchanged implementation of every statistical property of the estimator.
- [EEGLAB](https://eeglab.org/) and [clean_rawdata / ASR](https://eeglab.org/plugins/clean_rawdata/): EEG processing, cleaning, and manual ICA workflow. Original experiment revisions are unknown.
- NumPy, SciPy (`LinearOperator`, `lsmr`, filtering, Hilbert transform), pandas, Matplotlib, and seaborn supply numerical and visualization infrastructure.

The source mentions a single-motor-unit corticomuscular-coherence reference without a complete bibliographic record. Its exact citation, and the original multi-kernel method's source/ownership record, must be recovered rather than guessed. gCKC is mentioned only as future work, not as implemented functionality.

No external toolbox code was added and no new license was assigned. Private-portfolio figure permission does not establish unrestricted public redistribution rights.
