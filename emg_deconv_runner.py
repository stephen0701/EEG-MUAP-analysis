# %%

"""
final version for 20241225 (imported as module)
EMG preprocessing and multi-kernel deconvolution module.
- Can be imported and called from other scripts.
- Provides single-dataset and batch processing functions.
"""

import os
import json
import logging
import glob
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, detrend, butter

from tool_kits.EMG_class import EMG_DATA
from tool_kits.multi_kernel_deconv_new import multi_kernel_deconv

def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path

def configure_logger(logfile_path):
    """Reconfigure root logger to log both to file and console."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(logfile_path), logging.StreamHandler()])

class AlternativeProcessor:
    """
    Simple preprocessing pipeline:
    1. Detrend (remove linear trend)
    2. Apply notch filter (remove powerline noise)
    3. Apply highpass filter (remove low-frequency drift)
    4. Apply lowpass filter (remove high-frequency noise)
    """
    def __init__(self, fs=1000, notch_f0=60, hp_fc=40, lp_fc=450):
        self.fs = fs
        # Design notch filter
        self.b_notch, self.a_notch = iirnotch(notch_f0, 30, fs)
        # Design highpass filter
        nyq = 0.5 * fs
        hp_wn = hp_fc / nyq
        self.b_hp, self.a_hp = butter(4, hp_wn, btype='highpass', output='ba')
        # Design lowpass filter
        lp_wn = lp_fc / nyq
        self.b_lp, self.a_lp = butter(4, lp_wn, btype='lowpass', output='ba')

    def process(self, x):
        """Apply detrend, notch, highpass, lowpass filters."""
        y = detrend(x)
        y = filtfilt(self.b_notch, self.a_notch, y)
        y = filtfilt(self.b_hp, self.a_hp, y)
        y = filtfilt(self.b_lp, self.a_lp, y)
        return y

def process_single_dataset(
    dataset_path,
    muscles=None,
    fs=1000,
    modes=('single', 'multi'),
    save_results=True,
    verbose=True
):
    """
    Preprocess EMG and run multi-kernel deconvolution for a single dataset folder.
    Returns: dict of results per mode.
    """
    if muscles is None:
        muscles = ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF']
    preprocessor = AlternativeProcessor(fs=fs)
    emg = EMG_DATA(dataset_path)
    emg.read_emg_file(0)
    event_indices = emg.get_event_id_indices()
    if event_indices is None or len(event_indices) < 2:
        raise ValueError(f"No valid event_id indices found in {dataset_path}")
    first_event_index = int(event_indices[0])
    last_event_index = int(event_indices[-1])
    emg.our_data = emg.our_data.iloc[first_event_index:last_event_index+1].reset_index(drop=True)
    emg_timestamps = emg.our_data['Timestamp'].to_numpy(dtype=float)
    df0 = emg.our_data
    available = [m for m in muscles if m in df0.columns]
    if not available:
        raise ValueError(f"No expected channels found in {dataset_path}; available: {df0.columns.tolist()}")
    clean_signals = []
    for ch in available:
        arr = df0[ch].astype(float).values
        clean_signals.append(preprocessor.process(arr))
    clean_signals = np.column_stack(clean_signals)

    # NaN protection
    clean_df = pd.DataFrame(clean_signals, columns=available)
    if clean_df.isna().values.any():
        clean_df = clean_df.interpolate(axis=0).bfill().ffill().fillna(0.0)
    clean_signals = clean_df.to_numpy()

    results = {}
    top_save = os.path.join(dataset_path, 'multi-kernel-deconv')
    ensure_dir(top_save)
    for mode in modes:
        mode_dir = os.path.join(top_save, mode)
        ensure_dir(mode_dir)
        np.save(os.path.join(mode_dir, "emg_timestamps.npy"), emg_timestamps)
        if verbose:
            print(f"Running multi-kernel deconvolution ({mode}) ...")
        deconv_result = multi_kernel_deconv(
            clean_signals,
            fs,
            force_mode=mode,
            save_dir=mode_dir if save_results else None,
            max_iter=10,
            tol=1e-6,
        )
        # Save pickle summary
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pkl_path = os.path.join(mode_dir, f'results_{mode}_{ts}.pkl')
        with open(pkl_path, 'wb') as pf:
            pickle.dump(deconv_result, pf, protocol=pickle.HIGHEST_PROTOCOL)
        # Find .npz file
        npz_pattern = os.path.join(mode_dir, 'multi_kernel_deconv_results_*')
        npz_candidates = sorted([p for p in glob.glob(npz_pattern) if p.endswith('.npz')])
        npz_path = npz_candidates[-1] if npz_candidates else None
        results[mode] = {'pickle': pkl_path, 'npz': npz_path, 'n_channels': len(deconv_result)}
    return results

def process_all_datasets(
    parent_dir,
    muscles=None,
    fs=1000,
    modes=('single', 'multi'),
    save_results=True,
    verbose=True
):
    """
    Batch process all path2_* datasets in parent_dir.
    Returns: master summary dict.
    """
    pattern = os.path.join(parent_dir, 'path2*')
    data_dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not data_dirs:
        print(f"No path2* directories found under {parent_dir}")
        return {}

    master_results = {
        'meta': {
            'parent_dir': parent_dir,
            'fs': fs,
            'muscles': muscles if muscles else ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF'],
            'modes': modes,
            'timestamp': datetime.now().isoformat()
        },
        'datasets': {}
    }

    for data_dir in data_dirs:
        print(f"\n=== Processing dataset: {data_dir} ===")
        try:
            ds_results = process_single_dataset(
                data_dir,
                muscles=muscles,
                fs=fs,
                modes=modes,
                save_results=save_results,
                verbose=verbose
            )
            master_results['datasets'][os.path.basename(data_dir)] = {
                'success': True,
                'files': ds_results
            }
        except Exception as e:
            print(f"Error processing {data_dir}: {e}")
            master_results['datasets'][os.path.basename(data_dir)] = {
                'success': False,
                'error': str(e)
            }

    # Save master summary
    master_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_fn = os.path.join(parent_dir, f"master_results_summary_{master_ts}.json")
    with open(master_fn, 'w', encoding='utf-8') as mf:
        json.dump(master_results, mf, ensure_ascii=False, indent=2)
    print(f"All done. Master summary saved to {master_fn}")
    return master_results

# Optional CLI entry
def main():
    # Example usage for batch processing
    PARENT_DIR = '/home/ntk/exo/ExoskeletonPowerAsistance/simulation/mocap_EMG_EEG_data/20241225'
    process_all_datasets(PARENT_DIR)

if __name__ == "__main__":
    main()
# %%
