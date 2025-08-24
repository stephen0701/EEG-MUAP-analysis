# %%
'''
Check for the right data
from EMG_class import EMG_DATA
base_path = '/home/ntk/Desktop/20250807_test_MUAP/2025_08_07(2)'
emg = EMG_DATA(base_path)
emg.read_emg_file(0) #RTA
emg.read_emg_file(1) #LRF
emg.read_emg_file(2) #RRF
emg.read_emg_file(3) #RBF
emg.read_emg_file(4) #LBF
emg.read_emg_file(5) #RGA
emg.read_emg_file(6) #LTA
emg.read_emg_file(7) #LGA
'''
from EMG_class import EMG_DATA
base_path = '/home/ntk/Desktop/20241225EMG'
emg = EMG_DATA(base_path)
emg.read_emg_file(0) #10
emg.read_emg_file(1) #02
emg.read_emg_file(2) #06
emg.read_emg_file(3) #07
emg.read_emg_file(4) #05
emg.read_emg_file(5) #04
emg.read_emg_file(6) #01
emg.read_emg_file(7) #08
emg.read_emg_file(8) #03
emg.read_emg_file(9) #09
# %%
''' 
Test two different preprocessing pipeline on multi kernel deconvolution
'''
import os
import time
import pickle
import json
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, detrend, butter
from EMG_class import EMG_DATA
from multi_kernel_deconv_new import multi_kernel_deconv

# ---------- Helpers ----------
def butter_filter(order, cutoff, btype, fs):
    nyq = 0.5 * fs
    wn = [c/nyq for c in cutoff] if isinstance(cutoff, (list, tuple)) else cutoff/nyq
    # return b, a pair for filtfilt
    return butter(order, wn, btype=btype, output='ba')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ---------- Preprocessors ----------
class AlternativeProcessor:
    def __init__(self, fs=1000, notch_f0=60, bp_fc=(40,450)):
        self.fs = fs
        self.b_notch, self.a_notch = iirnotch(notch_f0, 30, fs)
        self.b_bp, self.a_bp = butter_filter(4, bp_fc, 'bandpass', fs)
    def process(self, x):
        y = detrend(x)
        y = filtfilt(self.b_notch, self.a_notch, y)
        y = filtfilt(self.b_bp, self.a_bp, y)
        return y

class EMDProcessor:
    def __init__(self, fs=1000, noise_window=(0,5)):
        self.fs = fs
        self.noise_start = int(noise_window[0]*fs)
        self.noise_end   = int(noise_window[1]*fs)
    def process(self, x):
        from emd.sift import ensemble_sift
        imfs = ensemble_sift(x, max_imfs=6, nensembles=24, nprocesses=6, ensemble_noise=0.2, imf_opts={'sd_thresh':0.1})
        noise_stds = np.std(imfs[self.noise_start:self.noise_end, :], axis=0)
        imf_thr = np.sign(imfs) * np.maximum(np.abs(imfs) - noise_stds, 0)
        return imf_thr.sum(axis=1)

# ---------- Single-file pipeline ----------
def process_file(fi, emg, muscles, preprocessors, modes, fs, base_save):
    file_key = f'file_{fi}'
    result = {'index': fi, 'runs': {}, 'error': None}
    try:
        df = emg.read_emg_file(fi)
        if df is None or (hasattr(df, 'empty') and df.empty):
            result['error'] = 'read_emg_file returned empty'
            return result

        available = [ch for ch in muscles if ch in df.columns]
        if not available:
            result['error'] = 'no target channels in file'
            return result

        for name, pre in preprocessors.items():
            run_info = {}
            save_base = ensure_dir(os.path.join(base_save, file_key, name))
            try:
                # process each available channel and stack
                arrs = []
                for ch in available:
                    arr = df[ch].astype(float).values
                    arrs.append(pre.process(arr))
                clean_signals = np.column_stack(arrs)  # (N, C)

                # interpolate NaNs (column-wise). Use numpy approach:
                if np.isnan(clean_signals).any():
                    xs = np.arange(clean_signals.shape[0])
                    for c in range(clean_signals.shape[1]):
                        col = clean_signals[:, c]
                        nans = np.isnan(col)
                        if nans.any():
                            clean_signals[nans, c] = np.interp(xs[nans], xs[~nans], col[~nans])

                for mode in modes:
                    mode_info = {'save_dir': None, 'result_path': None, 'error': None, 'start_time': None, 'end_time': None}
                    sd = ensure_dir(os.path.join(save_base, mode))
                    mode_info['save_dir'] = sd
                    mode_info['start_time'] = datetime.now().isoformat()
                    try:
                        start = time.time()
                        res = multi_kernel_deconv(clean_signals, fs, force_mode=mode, save_dir=sd)
                        dur = time.time() - start
                        rp = os.path.join(sd, 'result.pkl')
                        with open(rp, 'wb') as f:
                            pickle.dump(res, f)
                        mode_info['result_path'] = rp
                        logging.info(f"[file_{fi}] {name}/{mode} done in {dur:.1f}s")
                    except Exception as e:
                        logging.exception(f"Deconv failed for file {fi} {name} {mode}")
                        mode_info['error'] = str(e)
                    finally:
                        mode_info['end_time'] = datetime.now().isoformat()
                    run_info[mode] = mode_info
            except Exception as e:
                logging.exception(f"Preprocessing failed for file {fi} pre {name}")
                run_info['error'] = str(e)
            result['runs'][name] = run_info

    except Exception as e_all:
        logging.exception(f"Unexpected processing error file {fi}")
        result['error'] = str(e_all)
    return result

# ---------- Main ----------
def main():
    base_path = '/home/ntk/Desktop/20250807_test_MUAP/2025_08_07(2)'
    base_save = os.path.join(base_path, 'plots', 'batch_results')
    ensure_dir(base_save)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(base_save, 'batch_run.log')), logging.StreamHandler()])

    fs = 1000
    muscles = ['RTA']
    #muscles = ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF']

    preprocessors = {
        'alt': AlternativeProcessor(fs=fs),
        'emd': EMDProcessor(fs=fs)
    }
    modes = ['single','multi']
    emg = EMG_DATA(base_path)

    results = {'meta': {'base_path': base_path, 'fs': fs, 'muscles': muscles, 'preprocessors': list(preprocessors.keys()), 'modes': modes, 'timestamp': datetime.now().isoformat()},
               'files': {}}
    file_indices = [0]
    #file_indices = list(range(8))
    for fi in tqdm(file_indices, desc='files'):
        results['files'][f'file_{fi}'] = process_file(fi, emg, muscles, preprocessors, modes, fs, base_save)

    # save summary
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_fn = os.path.join(base_save, f"results_summary_{ts}.json")
    with open(summary_fn, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    logging.info("Batch complete. Summary saved to %s", summary_fn)

if __name__ == '__main__':
    main()
# %%
''' 
Final version for 20250807_test_MUAP
'''
import os
import time
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, detrend, butter
from EMG_class import EMG_DATA
from multi_kernel_deconv_new import multi_kernel_deconv

# ---------- Helpers ----------
def butter_filter(order, cutoff, btype, fs):
    nyq = 0.5 * fs
    wn = [c/nyq for c in cutoff] if isinstance(cutoff, (list, tuple)) else cutoff/nyq
    # return b, a pair for filtfilt
    return butter(order, wn, btype=btype, output='ba')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ---------- Preprocessors ----------
class AlternativeProcessor:
    def __init__(self, fs=1000, notch_f0=60, bp_fc=(40,450)):
        self.fs = fs
        self.b_notch, self.a_notch = iirnotch(notch_f0, 30, fs)
        self.b_bp, self.a_bp = butter_filter(4, bp_fc, 'bandpass', fs)
    def process(self, x):
        # detrend and apply notch + bandpass
        y = detrend(x)
        y = filtfilt(self.b_notch, self.a_notch, y)
        y = filtfilt(self.b_bp, self.a_bp, y)
        return y

# ---------- Single-file pipeline ----------
def process_file(fi, emg, muscles, preprocessors, modes, fs, base_save):
    """
    Process one EMG file index `fi` with the provided preprocessor(s) and deconv mode(s).
    - emg: EMG_DATA instance with method read_emg_file(fi) -> pandas.DataFrame
    - muscles: list of muscle/channel names expected (order matters)
    - preprocessors: dict name -> preprocessor object with .process(array) method
    - modes: list of strings (e.g., ['single','multi'])
    - fs: sampling rate (Hz)
    - base_save: base folder to write outputs (function will create file_{fi}/<mode>/result_<preproc>.npz)
    Returns a summary dict with status and per-run info.
    """
    file_key = f'file_{fi}'
    result = {'index': fi, 'runs': {}, 'error': None}
    try:
        # read EMG data for file index fi (EMG_class.EMG_DATA.read_emg_file)
        df = emg.read_emg_file(fi)
        if df is None or (hasattr(df, 'empty') and df.empty):
            result['error'] = 'read_emg_file returned empty'
            return result

        # determine which muscles/channels are present in this file (preserve order from `muscles`)
        available = [ch for ch in muscles if ch in df.columns]
        if not available:
            result['error'] = 'no target channels in file'
            return result

        # loop over preprocessors (e.g., 'alt')
        for name, pre in preprocessors.items():
            run_info = {}
            file_base = ensure_dir(os.path.join(base_save, file_key))
            try:
                # preprocess each available channel and stack into (N, C)
                arrs = []
                for ch in available:
                    arr = df[ch].astype(float).values
                    arrs.append(pre.process(arr))
                clean_signals = np.column_stack(arrs)  # shape (N, C)

                # interpolate NaNs robustly with pandas
                clean_df = pd.DataFrame(clean_signals, columns=available)
                if clean_df.isna().values.any():
                    clean_df = clean_df.interpolate(axis=0).bfill().ffill().fillna(0.0)
                clean_signals = clean_df.to_numpy()

                # process each requested deconv mode
                for mode in modes:
                    mode_info = {'save_dir': None, 'result_path': None, 'error': None, 'start_time': None, 'end_time': None}
                    sd = ensure_dir(os.path.join(file_base, mode))
                    mode_info['save_dir'] = sd
                    mode_info['start_time'] = datetime.now().isoformat()
                    try:
                        start = time.time()
                        # run your multi-kernel deconvolution; expects (N, C) and returns list-of-channel dicts
                        res = multi_kernel_deconv(clean_signals, fs, force_mode=mode, save_dir=sd)
                        dur = time.time() - start

                        # construct npz filename and prepare dict of arrays
                        out_base = os.path.join(sd, f'result_{name}')
                        npz_path = out_base + '.npz'
                        save_dict = {}

                        # channel name mapping: use available[] order (must match res order)
                        chan_names = available[:]  # copy list of channel names for mapping

                        # if mismatch in lengths, record a warning but still try to save
                        if len(res) != len(chan_names):
                            logging.warning(f"[file_{fi}] number of deconv channels ({len(res)}) != available channels ({len(chan_names)}). Will attempt best-effort saving.")
                            # if available shorter/longer, pad or truncate chan_names accordingly
                            if len(res) > len(chan_names):
                                # create placeholder names for missing entries
                                extra = [f"chan_extra_{i}" for i in range(len(chan_names), len(res))]
                                chan_names.extend(extra)
                            else:
                                chan_names = chan_names[:len(res)]

                        # populate save_dict per channel
                        for i, chres in enumerate(res):
                            prefix = f"chan{i}_"
                            muscle_name = chan_names[i] if i < len(chan_names) else f"chan{i}"

                            # save core numeric arrays if present
                            if chres.get('X') is not None:
                                save_dict[prefix + 'X'] = np.asarray(chres['X'])
                            if chres.get('K') is not None:
                                save_dict[prefix + 'K'] = np.asarray(chres['K'])
                            if chres.get('recon_total') is not None:
                                save_dict[prefix + 'recon_total'] = np.asarray(chres['recon_total'])
                            if chres.get('CWF') is not None:
                                save_dict[prefix + 'CWF'] = np.asarray(chres['CWF'])
                            if chres.get('CWF_comp') is not None:
                                save_dict[prefix + 'CWF_comp'] = np.asarray(chres['CWF_comp'])

                            # robust handling of recon_components: try to make a clean 2D array; otherwise store each component separately
                            recon_comps = chres.get('recon_components', None)
                            if recon_comps is not None:
                                try:
                                    arr_rc = np.asarray(recon_comps)
                                    if arr_rc.ndim == 2:
                                        # keep it
                                        save_dict[prefix + 'recon_components'] = arr_rc
                                    else:
                                        # try stack (list of 1D arrays -> (nK, N))
                                        stacked = np.stack(recon_comps, axis=0)
                                        save_dict[prefix + 'recon_components'] = stacked
                                except Exception:
                                    # fallback: save each component individually to avoid object dtype
                                    for kk, comp in enumerate(recon_comps):
                                        try:
                                            save_dict[f"{prefix}recon_k{kk}"] = np.asarray(comp)
                                        except Exception:
                                            # last resort: store text repr (unicode)
                                            save_dict[f"{prefix}recon_k{kk}_txt"] = np.array([repr(comp)], dtype='U')

                            # metadata: include muscle name + key numerical/meta fields
                            meta = {k: chres[k] for k in ('sigmas', 'mode', 'conv_history', 'alpha', 'iterations') if k in chres}
                            meta['muscle'] = muscle_name
                            try:
                                meta_json = json.dumps(meta, default=str)
                            except Exception:
                                meta_json = json.dumps({k: str(meta.get(k)) for k in meta})
                            save_dict[prefix + 'meta_json'] = np.array([meta_json], dtype='U')

                        # global metadata and channel-name list (for direct mapping)
                        global_meta = {
                            'file_index': fi,
                            'preprocessor': name,
                            'mode': mode,
                            'fs': fs,
                            'timestamp': datetime.now().isoformat()
                        }
                        save_dict['__global_meta_json__'] = np.array([json.dumps(global_meta)], dtype='U')
                        save_dict['__chan_names_json__'] = np.array([json.dumps(chan_names)], dtype='U')

                        # write compressed npz (safe)
                        try:
                            np.savez_compressed(npz_path, **save_dict)
                            mode_info['result_path'] = npz_path
                        except Exception as e_save:
                            logging.exception(f"Failed to save npz {npz_path}: {e_save}")
                            mode_info['error'] = f"save_failed: {e_save}"

                        # record duration
                        logging.info(f"[file_{fi}] {name}/{mode} done in {dur:.1f}s")
                    except Exception as e_run:
                        # if multi_kernel_deconv or saving failed
                        logging.exception(f"Deconv failed for file {fi} pre={name} mode={mode}: {e_run}")
                        mode_info['error'] = str(e_run)
                    finally:
                        mode_info['end_time'] = datetime.now().isoformat()
                    run_info[mode] = mode_info
            except Exception as e_pre:
                logging.exception(f"Preprocessing failed for file {fi} pre {name}: {e_pre}")
                run_info['error'] = str(e_pre)
            result['runs'][name] = run_info

    except Exception as e_all:
        logging.exception(f"Unexpected processing error file {fi}: {e_all}")
        result['error'] = str(e_all)
    return result

# ---------- Main ----------
def main():
    base_path = '/home/ntk/Desktop/20250807_test_MUAP/2025_08_07(2)'
    base_save = os.path.join(base_path, 'plots', 'final_results')
    ensure_dir(base_save)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(base_save, 'batch_run.log')), logging.StreamHandler()])

    fs = 1000
    muscles = ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF']

    # only 'alt' preprocessor now
    preprocessors = {
        'alt': AlternativeProcessor(fs=fs),
    }
    modes = ['single','multi']
    emg = EMG_DATA(base_path)
  
    results = {'meta': {'base_path': base_path, 'fs': fs, 'muscles': muscles, 'preprocessors': list(preprocessors.keys()), 'modes': modes, 'timestamp': datetime.now().isoformat()},
               'files': {}}

    # if only run file 0, change file_indices = [0]
    file_indices = list(range(8))
    total = len(file_indices)
    try:
        for idx, fi in enumerate(file_indices, start=1):
            logging.info(f"Processing file {idx}/{total}: file_{fi}")
            results['files'][f'file_{fi}'] = process_file(fi, emg, muscles, preprocessors, modes, fs, base_save)
    except KeyboardInterrupt:
        logging.warning('Processing interrupted by user')

    # save summary
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_fn = os.path.join(base_save, f"results_summary_{ts}.json")
    with open(summary_fn, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    logging.info("Batch complete. Summary saved to %s", summary_fn)

if __name__ == '__main__':
    main()
# %%
"""
final version for 20241225 (updated saving layout)
- plots -> <data_dir>/multi-kernel-deconv/<mode>/plots/
- aggregated npz -> <data_dir>/multi-kernel-deconv/<mode>/multi_kernel_deconv_results_<mode>_<ts>.npz
- pickle summary -> <data_dir>/multi-kernel-deconv/<mode>/results_<mode>_<ts>.pkl
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

from EMG_class import EMG_DATA
from multi_kernel_deconv_new import multi_kernel_deconv

# ---------- small helpers ----------
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
    3. Apply bandpass filter (retain frequency band of interest)
    """
    def __init__(self, fs=1000, notch_f0=60, bp_fc=(40, 450)):
        self.fs = fs
        # Design notch filter
        self.b_notch, self.a_notch = iirnotch(notch_f0, 30, fs)
        # Design bandpass filter
        nyq = 0.5 * fs
        wn = [c/nyq for c in bp_fc]
        self.b_bp, self.a_bp = butter(4, wn, btype='bandpass', output='ba')

    def process(self, x):
        """Apply detrend, notch, and bandpass filters to signal x."""
        y = detrend(x)
        y = filtfilt(self.b_notch, self.a_notch, y)
        y = filtfilt(self.b_bp, self.a_bp, y)
        return y

def main():
    # Parent dir containing path2_* subfolders
    PARENT_DIR = '/home/ntk/Desktop/20241225'
    pattern = os.path.join(PARENT_DIR, 'path2*')
    data_dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not data_dirs:
        print(f"No path2* directories found under {PARENT_DIR}")
        return

    # Settings
    fs = 1000
    muscles = ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF']
    preprocessors = {'alt': AlternativeProcessor(fs=fs)}
    modes = ['single', 'multi']

    master_results = {
        'meta': {
            'parent_dir': PARENT_DIR,
            'fs': fs,
            'muscles': muscles,
            'preprocessors': list(preprocessors.keys()),
            'modes': modes,
            'timestamp': datetime.now().isoformat()
        },
        'datasets': {}
    }

    # iterate datasets
    for data_dir in data_dirs:
        print(f"\n=== Processing dataset: {data_dir} ===")
        top_save = os.path.join(data_dir, 'multi-kernel-deconv')   # top folder requested
        ensure_dir(top_save)

        # create a dataset-level log file in top_save
        logfile = os.path.join(top_save, 'batch_run.log')
        configure_logger(logfile)

        ds_summary = {'data_dir': data_dir, 'success': False, 'files': {}, 'error': None}
        try:
            emg = EMG_DATA(data_dir)

            # Align EMG data to event ids (use your EMG class methods)
            emg.read_emg_file(0)            # read the single EMG file
            emg.display_event_id_data()     # optionally print event info
            event_indices = emg.get_event_id_indices()
            if event_indices is None or len(event_indices) < 2:
                logging.warning(f"No valid event_id indices found in {data_dir}")
                ds_summary['error'] = 'no_event_indices'
                master_results['datasets'][os.path.basename(data_dir)] = ds_summary
                continue
            first_event_index = int(event_indices[0])
            last_event_index = int(event_indices[-1])
            emg.our_data = emg.our_data.iloc[first_event_index:last_event_index+1].reset_index(drop=True)
            # extract timestamp column for time(s)
            emg_timestamps = emg.our_data['Timestamp'].to_numpy(dtype=float)

            df0 = emg.our_data
            if df0 is None or (hasattr(df0, 'empty') and df0.empty):
                logging.warning(f"EMG data empty after alignment for {data_dir}")
                ds_summary['error'] = 'empty_after_alignment'
                master_results['datasets'][os.path.basename(data_dir)] = ds_summary
                continue

            # pick available muscle channels in the expected order
            available = [m for m in muscles if m in df0.columns]
            if not available:
                logging.warning(f"No expected channels found in {data_dir}; available: {df0.columns.tolist()}")
                ds_summary['error'] = 'no_expected_channels'
                master_results['datasets'][os.path.basename(data_dir)] = ds_summary
                continue

            # preprocess channels into shape (N, C_actual)
            arrs = []
            for ch in available:
                arr = df0[ch].astype(float).values
                pre = preprocessors['alt']
                arrs.append(pre.process(arr))
            clean_signals = np.column_stack(arrs)

            # NaN protection
            clean_df = pd.DataFrame(clean_signals, columns=available)
            if clean_df.isna().values.any():
                clean_df = clean_df.interpolate(axis=0).bfill().ffill().fillna(0.0)
            clean_signals = clean_df.to_numpy()

            ds_results = {}
            # for each mode create a separate folder and call deconv
            for mode in modes:
                logging.info(f"Processing dataset {os.path.basename(data_dir)} mode={mode}")
                # mode_dir: e.g., <data_dir>/multi-kernel-deconv/single or /multi
                mode_dir = os.path.join(top_save, mode)
                ensure_dir(mode_dir)

                try:
                    # save emg_timestamps in this mode folder
                    ts_path = os.path.join(mode_dir, "emg_timestamps.npy")
                    np.save(ts_path, emg_timestamps)
                    # call multi_kernel_deconv
                    # IMPORTANT: multi_kernel_deconv should save PNGs under mode_dir/plots/
                    # and the aggregated .npz under mode_dir/
                    res = multi_kernel_deconv(
                        clean_signals,
                        fs,
                        force_mode=mode,
                        save_dir=mode_dir,            
                        max_iter=10,
                        tol=1e-6,
                    )

                    # gather generated .npz file (pick the newest matching name in mode_dir)
                    npz_pattern = os.path.join(mode_dir, 'multi_kernel_deconv_results_*')
                    npz_candidates = sorted([p for p in glob.glob(npz_pattern) if p.endswith('.npz')])
                    npz_path = npz_candidates[-1] if npz_candidates else None

                    # Save a pickle summary (store in mode_dir alongside .npz)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    pkl_path = os.path.join(mode_dir, f'results_{mode}_{ts}.pkl')
                    with open(pkl_path, 'wb') as pf:
                        pickle.dump(res, pf, protocol=pickle.HIGHEST_PROTOCOL)

                    ds_results[mode] = {'pickle': pkl_path, 'npz': npz_path, 'n_channels': len(res)}
                    logging.info(f"Mode {mode} done. pickle={pkl_path}, npz={npz_path}")

                except Exception as e_mode:
                    logging.exception(f"Deconvolution failed for {data_dir} mode={mode}: {e_mode}")
                    ds_results[mode] = {'error': str(e_mode)}
                    continue

            ds_summary['success'] = True
            ds_summary['files'] = ds_results
            master_results['datasets'][os.path.basename(data_dir)] = ds_summary

        except KeyboardInterrupt:
            logging.warning('Processing interrupted by user')
            ds_summary['error'] = 'interrupted'
            master_results['datasets'][os.path.basename(data_dir)] = ds_summary
        except Exception as e_all:
            logging.exception(f"Top-level failure processing {data_dir}: {e_all}")
            ds_summary['error'] = str(e_all)
            master_results['datasets'][os.path.basename(data_dir)] = ds_summary

    # save master summary
    master_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_fn = os.path.join(PARENT_DIR, f"master_results_summary_{master_ts}.json")
    with open(master_fn, 'w', encoding='utf-8') as mf:
        json.dump(master_results, mf, ensure_ascii=False, indent=2)
    print(f"All done. Master summary saved to {master_fn}")


if __name__ == "__main__":
    main()
# %%
