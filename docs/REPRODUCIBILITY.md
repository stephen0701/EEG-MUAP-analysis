# Reproduction guide

## Status and installation

The repository exposes research functions and executable examples. It contains no `eeg_emg_pipeline.py`, packaged CLI, or participant dataset. Report results have not been reproduced end to end during portfolio preparation.

From the root, run `python -m venv .venv`. Activate it (PowerShell: `.venv\Scripts\Activate.ps1`; bash: `source .venv/bin/activate`), then run `python -m pip install -r requirements.txt`. Requirements reflect imports, not the recovered original experiment environment. See [Verification](VERIFICATION.md) for the checked environment.

MATLAB EEG preprocessing requires Signal Processing Toolbox, DSP System Toolbox, EEGLAB/ASR, manual ICA, and the acquisition montage. Set `RESEARCH_CHANNEL_LOCATIONS` or `EEGPreprocessor.channel_location_file` to the actual 32-channel `.ced` file. Experiment toolbox versions are unknown.

## Private input contract

```text
private-data/path2_<trial>/
  EEG/<preprocessed>_align.csv
  EMG/<processed>_align.csv
  opensim/<motion>.trc
  movement_analysis/movement_results_*.pkl
  multi-kernel-deconv/{single,multi}/results_*.pkl
  firing_pattern_to_spikes/spike_results_*.pkl
  neural_movement_segmentation/segments_*.pkl
```

This is a contract illustration, not example data. Original EEG/EMG exports are also required for the initial alignment route.

- **MATLAB raw EEG:** `EEG_path2_*.csv`, channel columns 3-34, event fields 35/38, events 130/141.
- **Raw EMG (`EMG_DATA`):** device-specific export with 10 metadata/header rows and comma-separated contents, `Timestamp`, event, and muscle fields; not a generic CSV reader.
- **Aligned EMG:** comma-separated numeric LGA, RGA, LTA, RTA, LBF, RBF, LRF, RRF; LGL/RGL aliases map to LGA/RGA. Default fs = 1000 Hz.
- **Aligned EEG:** time/Timestamp plus electrode columns, default fs = 500 Hz. Synthesizing missing timestamps is a fallback, not alignment verification.
- **Motion:** tab-separated TRC, five header rows, frame/time fields, and source-specific marker order. The v2 detector also uses synchronization markers and is not a drop-in replacement for alignment.
- **Products:** nested dictionaries saved in pickle. Load only trusted local pickles; no portable dataset standard is claimed.

Several loaders take the first glob match. Keep only the intended version of each artifact in a trial folder and record selected filenames. Examples write products into trial folders; use a working copy of authorized data outside the checkout.

## Function-level usage

These calls were inspected, not validated as a complete report reproduction. Run from the repository root. After EEG preprocessing and verified motion alignment have created `*_align.csv`:

```python
import os
from tool_kits.EMG_to_MUAP import process_single_dataset
from tool_kits.firing_pattern_to_spikes import process_single_dataset_spikes

dataset = os.environ['RESEARCH_DATASET']
deconvolution = process_single_dataset(dataset, fs=1000, modes=('single', 'multi'))
spikes = process_single_dataset_spikes(
    dataset, fs=1000.0,
    spike_params={
        'threshold': None, 'use_percentile': 97, 'k_mad': 3.0,
        'smooth_sigma_ms': 0.0, 'min_isi_ms': 30.0,
        'min_prominence': None,
    },
    exclude_time_range=None,  # Record the experiment-specific choice.
)
```

The existing batch alignment route is `movement_state_detector.IntegratedMovementStateSegmentor.process_dataset_transitions(rootFolder=..., prefix='path2_')`. Its rules and device-specific readers need validation on the actual acquisition.

After verifying phase timing and required products:

```python
from tool_kits.neural_data_movement_segmentor import segment_single_dataset
segments = segment_single_dataset(dataset)
```

Choose the intended saved `segments_*.pkl` explicitly, then provide caller-selected local input/output paths:

```python
from tool_kits.coherence_analysis import analyze_single_dataset_coherence
from tool_kits.spike_triggered_analysis import analyze_single_dataset_correlations

coherence = analyze_single_dataset_coherence(segments_file_path, coherence_output_dir)
sta_ppc = analyze_single_dataset_correlations(segments_file_path, sta_ppc_output_dir)
```

`movement_state_detector_v2.py` and `spike_triggered_analysis_new.py` are exploratory alternatives, not established final report routes.

## Executable example configuration

| Variable | Use |
| --- | --- |
| `RESEARCH_DATASET` | `python -m tool_kits.EMG_to_MUAP` |
| `RESEARCH_DATA_ROOT` | Other Python examples; several select conventional trial `path2_02` below this root |
| `RESEARCH_EEG_INPUT` | MATLAB example input CSV folder |
| `RESEARCH_CHANNEL_LOCATIONS` | Actual acquisition montage |
| `RESEARCH_ICA_DIR` | Reviewed ICA `.set` folder |
| `RESEARCH_EEG_OUTPUT` | MATLAB z-scored EEG export folder |

These examples do not parse `--help` or dataset CLI flags. Missing required environment variables raise an error before data loading. API calls accept explicit dataset paths. [Methods](METHODS.md) explains coherence, PPC, STA axes, exclusions, and remaining numerical edge cases.
