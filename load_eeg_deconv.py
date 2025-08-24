# %%
# load_eeg_deconv.py
"""
Read EEG CSV files and aggregated multi-kernel-deconv .npz results.
This module is read-only (does not save files). It provides utility
functions to load EEG csvs (time + channel data) and to locate + parse
the aggregated npz produced by multi_kernel_deconv.

Usage:
    from load_eeg_and_npz import load_all, load_eeg_from_folder, load_npz_summary
    master = load_all('/home/ntk/Desktop/20241225', modes=('single','multi'))
"""

import os
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------
# CSV reading utilities
# ---------------------------
def try_read_csv(path):
    """
    Try common separators to read a CSV file and return a pandas DataFrame.
    Tries ',', ';', and '\t' first; falls back to pandas default parsing.
    """
    separators = [",", ";", "\t"]
    for sep in separators:
        try:
            df = pd.read_csv(path, sep=sep, header=0, engine='python', encoding='utf-8')
            # require at least two columns (time + at least one channel)
            if df.shape[1] >= 2:
                return df
        except Exception:
            # try next separator
            pass
    # fallback: let pandas try its default parsing (may raise)
    return pd.read_csv(path, header=0, engine='python', encoding='utf-8')


def load_eeg_from_folder(folder):
    """
    Load all CSV files in `folder`. Each CSV is expected to have:
      - first column: time in seconds
      - remaining columns: channels (numeric)

    Returns a dictionary keyed by CSV basename with values:
      {
        'time': np.ndarray of shape (N,),
        'data': np.ndarray of shape (N, n_channels),
        'channels': list of channel names (header columns excluding time),
        'fs_est': estimated sample rate (Hz) or None if irregular,
        'path': full file path
      }

    The function prints a short status message per file.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    out = {}
    if not files:
        return out

    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = try_read_csv(fp)
        except Exception as e:
            print(f"[ERROR] reading CSV {fp}: {e}")
            continue

        cols = df.columns.tolist()
        if len(cols) < 2:
            print(f"[WARN] CSV {fp} has fewer than 2 columns; skipping")
            continue

        try:
            # convert to float arrays; this will raise on non-numeric content
            eeg_timestamps = df.iloc[:, 0].to_numpy(dtype=float)
            data = df.iloc[:, 1:].to_numpy(dtype=float)
        except Exception as e:
            print(f"[ERROR] converting CSV values to float for {fp}: {e}")
            continue

        channels = cols[1:]
        dt = np.diff(eeg_timestamps)
        # estimate sampling rate if time spacing is uniform and non-zero
        if dt.size > 0 and np.allclose(dt, dt[0], rtol=1e-3, atol=1e-6) and dt[0] != 0:
            fs_est = float(1.0 / dt[0])
        else:
            fs_est = None

        out[base] = {
            'time': eeg_timestamps,
            'data': data,
            'channels': channels,
            'fs_est': fs_est,
            'path': fp
        }
        n_samples, n_ch = data.shape
        print(f"[OK] EEG {base}: samples={n_samples}, channels={n_ch}, fs_est={fs_est}")
    return out

# ---------------------------
# NPZ reading utilities
# ---------------------------
def _try_decode_json_str(s):
    """
    Try to decode a JSON string into Python object. On failure return original string.
    Useful because the npz writer sometimes stores JSON-as-string in a one-element unicode array.
    """
    try:
        return json.loads(s)
    except Exception:
        return s


def load_npz_summary(npz_path):
    """
    Load and parse an aggregated multi-kernel-deconv .npz file.
    Returns a dict with:
      {
        'path': npz_path,
        'meta': parsed global meta (if present) or None,
        'n_channels': int or None (parsed from __n_channels__),
        'channels': { label: {varname: value_or_array, ...}, ... },
        'raw_keys': list of all keys present in the npz
      }

    Note: values that were stored as simple JSON/text single-element arrays are
    decoded where possible. Numeric arrays are returned as numpy arrays.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    result = {'path': npz_path, 'meta': None, 'n_channels': None, 'channels': {}, 'raw_keys': []}

    # use context manager to ensure file handle closure
    with np.load(npz_path, allow_pickle=True) as D:
        keys = list(D.keys())
        result['raw_keys'] = keys

        # read global meta if present (two possible keys used by saver)
        if '__global_meta_json__' in D:
            try:
                raw = D['__global_meta_json__'].ravel()[0]
                if isinstance(raw, (np.ndarray, np.generic)):
                    raw = raw.tolist()
                # try to parse JSON string
                result['meta'] = json.loads(str(raw))
            except Exception:
                try:
                    result['meta'] = str(D['__global_meta_json__'].ravel()[0])
                except Exception:
                    result['meta'] = None
        elif '__global_meta_txt__' in D:
            try:
                result['meta'] = str(D['__global_meta_txt__'].ravel()[0])
            except Exception:
                result['meta'] = None

        # parse channel count if present
        if '__n_channels__' in D:
            try:
                result['n_channels'] = int(np.asarray(D['__n_channels__']).ravel()[0])
            except Exception:
                result['n_channels'] = None

        # parse other keys; expected naming convention: "<label>_<varname>"
        for k in keys:
            if k.startswith('__'):
                # skip metadata keys already handled
                continue
            if '_' in k:
                label, var = k.split('_', 1)
            else:
                label, var = 'global', k

            if label not in result['channels']:
                result['channels'][label] = {}

            val = D[k]
            # If this is a single-element textual array, attempt JSON decode
            if isinstance(val, np.ndarray) and val.dtype.kind in ('U', 'S') and val.size == 1:
                s = str(val.ravel()[0])
                decoded = _try_decode_json_str(s)
                result['channels'][label][var] = decoded
            else:
                # keep numpy arrays (or pickled objects) as-is
                result['channels'][label][var] = val

    return result


def find_npz_in_mode_folder(mode_folder):
    """
    Find the aggregated npz file matching pattern:
      multi_kernel_deconv_results_*.npz
    Return the most recently modified match (latest), or None if none found.
    """
    pattern = os.path.join(mode_folder, 'multi_kernel_deconv_results_*.npz')
    matches = glob.glob(pattern)
    if not matches:
        return None
    # sort by modification time and return the newest
    matches = sorted(matches, key=lambda p: os.path.getmtime(p))
    return matches[-1]


# ---------------------------
# Top-level loader
# ---------------------------
def load_all(parent_dir, modes=('single', 'multi')):
    """
    Scan parent_dir for subfolders matching 'path2*'. For each dataset:
      - load EEG CSVs from <data_dir>/EEG (first column=time, rest channels)
      - find aggregated npz under <data_dir>/multi-kernel-deconv/<mode>/ and parse it

    Returns a dictionary:
      master[dataset_basename] = {
          'path': data_dir,
          'eeg': { csv_basename: {...}, ... },   # output of load_eeg_from_folder
          'npz': { mode: loaded_npz_dict_or_None, ... },
          'errors': [ ... ]
      }
    """
    master = {}
    pattern = os.path.join(parent_dir, 'path2*')
    data_dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not data_dirs:
        print(f"[WARN] no datasets (path2*) found under {parent_dir}")
        return master

    for data_dir in data_dirs:
        base = os.path.basename(data_dir)
        print(f"\n[INFO] dataset: {base}")
        entry = {'path': data_dir, 'eeg': {}, 'npz': {}, 'errors': []}

        # load EEGs from the EEG subfolder if it exists
        eeg_folder = os.path.join(data_dir, 'EEG')
        if os.path.isdir(eeg_folder):
            try:
                eegs = load_eeg_from_folder(eeg_folder)
                entry['eeg'] = eegs
                if not eegs:
                    print(f"[WARN] no CSV files found in {eeg_folder}")
            except Exception as e:
                entry['errors'].append(f"eeg_load_error:{str(e)}")
                print(f"[ERROR] loading EEG for {base}: {e}")
        else:
            print(f"[WARN] EEG folder not found at {eeg_folder}")

        # locate aggregated npz files under multi-kernel-deconv/<mode>/
        top_save = os.path.join(data_dir, 'multi-kernel-deconv')
        for mode in modes:
            mode_folder = os.path.join(top_save, mode)
            if not os.path.isdir(mode_folder):
                entry['npz'][mode] = None
                print(f"[WARN] mode folder missing: {mode_folder}")
                continue

            npz_path = find_npz_in_mode_folder(mode_folder)
            if npz_path is None:
                entry['npz'][mode] = None
                print(f"[WARN] no npz found in {mode_folder}")
                continue

            try:
                npz_loaded = load_npz_summary(npz_path)
                entry['npz'][mode] = npz_loaded
                print(f"[OK] loaded npz for mode {mode}: {os.path.basename(npz_path)}")
            except Exception as e:
                entry['npz'][mode] = None
                entry['errors'].append(f"npz_load_error_{mode}:{str(e)}")
                print(f"[ERROR] loading npz {npz_path}: {e}")

        master[base] = entry

    return master


def print_summary(master):
    """
    Print a short human-readable summary of the master dictionary.
    Useful for quick checks after calling load_all().
    """
    print("\n==== SUMMARY ====")
    for name, entry in master.items():
        print(f"Dataset: {name}")
        print(f"  path: {entry.get('path')}")
        eegs = entry.get('eeg', {})
        if eegs:
            print(f"  EEG CSVs: {list(eegs.keys())}")
            for csvname, info in eegs.items():
                print(f"    {csvname}: channels={info['channels']}, fs_est={info['fs_est']}, samples={info['data'].shape[0]}")
        else:
            print("  EEG CSVs: (none)")

        npzs = entry.get('npz', {})
        for mode, info in npzs.items():
            if info is None:
                print(f"  NPZ [{mode}]: (none)")
            else:
                labels = list(info.get('channels', {}).keys())
                nchan = info.get('n_channels')
                print(f"  NPZ [{mode}]: file={os.path.basename(info['path'])}, n_channels_field={nchan}, labels={labels}")
        if entry.get('errors'):
            print(f"  Errors: {entry['errors']}")
        print("------------------")


# ---------------------------
# Quick CLI test when run directly
# ---------------------------
if __name__ == "__main__":
    # Example: change PARENT_DIR to your actual directory
    PARENT_DIR = '/home/ntk/Desktop/20241225'
    master = load_all(PARENT_DIR, modes=('single', 'multi'))
    print_summary(master)

# %%
