# %%
"""
Convert deconvolution firing patterns (coefficients per sample) to spike times.
Designed to work with the multi-dataset structure and save results in individual dataset folders.

Main steps:
1. Optional Gaussian smoothing (reduce high-frequency noise)
2. Automatic or manual threshold determination (median + k*MAD or percentile)
3. Use scipy.signal.find_peaks for peak detection (height/prominence)
4. Use non-maximum suppression (based on amplitude) to enforce minimum ISI (refractory)
5. Generate spike_indices (samples), spike_times (s), binary vector, and return statistics & plots
6. Process each column of firing pattern array separately for multi-component analysis

Default for muscle MU: fs = 1000 Hz, min_isi_ms = 30 ms (modifiable)
References:
- Quiroga et al., 2004 (spike detection ideas using robust thresholds)
- Holobar & Zazula; Negro, Holobar & Farina et al. EMG decomposition literature (peak-based spike extraction)
- MU firing rate physiology references: De Luca et al.
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import glob
import pickle
from typing import Tuple, Optional, Dict, Any, Union, List
import warnings

def mad(x: np.ndarray) -> float:
    """Median absolute deviation (robust scale estimator, normalized for Gaussian)."""
    return np.median(np.abs(x - np.median(x))) / 0.6745



def preprocess_firing_pattern(firing_pattern: np.ndarray,
                              smooth_sigma: float = 0.0) -> np.ndarray:
    """
    Smooth/preprocess firing pattern.
    
    Args:
        firing_pattern: Input firing pattern array
        smooth_sigma: Gaussian sigma in samples (not seconds). If 0, no smoothing applied.
        
    Returns:
        Preprocessed firing pattern array
    """
    arr = np.asarray(firing_pattern).astype(float)
    if smooth_sigma is None or smooth_sigma <= 0:
        return arr.copy()
    # gaussian_filter1d sigma is in samples
    return gaussian_filter1d(arr, sigma=smooth_sigma, mode='nearest')

def auto_threshold_by_mad(arr: np.ndarray, k: float = 3.0) -> float:
    """
    Use robust median + k*MAD to determine threshold (recommended for asymmetric noise).
    
    Args:
        arr: Input array
        k: Multiplier for MAD (typically 3-5)
        
    Returns:
        Computed threshold value
    """
    med = np.median(arr)
    dev = mad(arr)
    return float(med + k * dev)

def detect_peaks_with_params(arr: np.ndarray,
                             height: Optional[float] = None,
                             prominence: Optional[float] = None,
                             distance_samples: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Use scipy.signal.find_peaks for peak detection, return peak indices and properties.
    
    Args:
        arr: Input array
        height: Minimum peak height
        prominence: Minimum peak prominence
        distance_samples: Initial peak spacing constraint (actual refractory enforced by NMS)
        
    Returns:
        Tuple of (peak_indices, peak_properties)
    """
    peaks, props = signal.find_peaks(arr, height=height, prominence=prominence, distance=distance_samples)
    return peaks, props

def enforce_min_isi_nms(peaks: np.ndarray,
                        arr: np.ndarray,
                        min_samples: int) -> np.ndarray:
    """
    Non-maximum suppression based on amplitude: sort peaks by amplitude,
    keep from largest to smallest, suppressing other peaks within min_samples range.
    
    Args:
        peaks: Peak indices array
        arr: Original array for amplitude comparison
        min_samples: Minimum samples between kept peaks
        
    Returns:
        Filtered peak indices with minimum ISI enforced
    """
    if len(peaks) == 0:
        return np.array([], dtype=int)
    
    # Sort peaks by amplitude descending
    order = np.argsort(arr[peaks])[::-1]
    peaks_sorted = peaks[order]
    keep_mask = np.zeros(len(arr), dtype=bool)
    kept = []
    suppressed = []
    
    for p in peaks_sorted:
        if not keep_mask[p]:
            kept.append(int(p))
            start = max(0, p - min_samples)
            end = min(len(arr), p + min_samples + 1)
            keep_mask[start:end] = True
        else:
            suppressed.append(int(p))
    
    kept_sorted = np.array(sorted(kept), dtype=int)
    suppressed_sorted = np.array(sorted(suppressed), dtype=int)
    return kept_sorted, suppressed_sorted

def convert_firing_to_spikes(firing_pattern: np.ndarray,
                            fs: float,
                            threshold: Optional[float] = None,
                            use_percentile: Optional[float] = None,
                            k_mad: float = 3.0,
                            smooth_sigma_ms: float = 0.0,
                            min_isi_ms: float = 30.0,
                            min_prominence: Optional[float] = None,
                            return_binary: bool = True,
                            exclude_time_range: Optional[Tuple[float, float]] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Convert firing pattern to spike indices and times.
    
    Args:
        firing_pattern: 1D array (coefficients per sample)
        fs: Sampling rate (Hz)
        threshold: If provided, use as height threshold; if None and use_percentile provided, 
                  use percentile; otherwise use median + k*MAD
        use_percentile: If provided (e.g., 97), use that percentile as threshold
        k_mad: k for median + k*MAD method
        smooth_sigma_ms: Gaussian smoothing sigma in milliseconds (0 = no smoothing)
        min_isi_ms: Minimum ISI (ms), default 30 ms (reasonable for MU)
        min_prominence: find_peaks prominence parameter (None for automatic)
        return_binary: Whether to return binary vector
        exclude_time_range: Optional (start_time, end_time) in seconds to exclude spikes
                          e.g., (0, 5) to exclude baseline period
        
    Returns:
        spike_indices (samples), spike_times_s (seconds), binary_vector (0/1), threshold_used, suppressed_indices
    """
    arr = np.asarray(firing_pattern).astype(float)
    n = arr.shape[0]

    # Convert sigma from ms to samples
    if smooth_sigma_ms is None:
        smooth_sigma_ms = 0.0
    smooth_sigma_samples = (smooth_sigma_ms / 1000.0) * fs
    if smooth_sigma_samples < 0.5:
        smooth_sigma_samples = 0.0

    processed = preprocess_firing_pattern(arr, smooth_sigma=smooth_sigma_samples)

    # Determine threshold
    if threshold is not None:
        thr = float(threshold)
    elif use_percentile is not None:
        thr = float(np.percentile(processed, float(use_percentile)))
    else:
        thr = float(auto_threshold_by_mad(processed, k=k_mad))

    # Set default prominence if not given: use fraction of (max - median)
    if min_prominence is None:
        dynamic_range = processed.max() - np.median(processed)
        # Choose conservative small prominence relative to range
        min_prominence = max(1e-6, dynamic_range * 0.1)

    # Initial peak detection
    # Distance param set to 1; actual refractory enforced by NMS below
    raw_peaks, props = detect_peaks_with_params(processed, height=thr, 
                                               prominence=min_prominence, distance_samples=1)

    # Enforce minimum ISI by NMS (non-maximum suppression)
    min_samples = max(1, int(round(min_isi_ms * fs / 1000.0)))
    spike_indices, suppressed_indices = enforce_min_isi_nms(raw_peaks, processed, min_samples=min_samples)

    # Filter out spikes in excluded time range (e.g., baseline period)
    if exclude_time_range is not None:
        start_time, end_time = exclude_time_range
        spike_times_temp = spike_indices.astype(float) / float(fs)
        
        # Find spikes within valid time range (outside excluded range)
        valid_mask = (spike_times_temp < start_time) | (spike_times_temp >= end_time)
        spike_indices = spike_indices[valid_mask]
        
        # Count excluded spikes for reporting
        n_excluded = np.sum(~valid_mask)
        if n_excluded > 0:
            print(f"      Excluded {n_excluded} spikes in time range [{start_time:.1f}s - {end_time:.1f}s]")
    
    # Binary vector
    binary_vec = np.zeros(n, dtype=int)
    if return_binary:
        binary_vec[spike_indices] = 1

    spike_times_s = spike_indices.astype(float) / float(fs)

    return spike_indices, spike_times_s, binary_vec, thr, suppressed_indices

def analyze_spike_statistics(spike_times_s: np.ndarray) -> Dict[str, float]:
    """
    Calculate spike statistics (count, mean_rate, mean_isi_ms, std_isi_ms, cv).
    
    Args:
        spike_times_s: Array of spike times in seconds
        
    Returns:
        Dictionary containing spike train statistics
    """
    stats: Dict[str, float] = {}
    if spike_times_s is None or len(spike_times_s) == 0:
        stats['spike_count'] = 0
        stats['mean_rate_hz'] = 0.0
        stats['mean_isi_ms'] = 0.0
        stats['std_isi_ms'] = 0.0
        stats['cv_isi'] = 0.0
        return stats

    spike_count = len(spike_times_s)
    stats['spike_count'] = float(spike_count)

    if spike_count >= 2:
        total_time = float(spike_times_s[-1] - spike_times_s[0])
        stats['mean_rate_hz'] = ((spike_count - 1) / total_time) if total_time > 0 else 0.0
        isi_ms = np.diff(spike_times_s) * 1000.0
        stats['mean_isi_ms'] = float(np.mean(isi_ms))
        stats['std_isi_ms'] = float(np.std(isi_ms))
        stats['cv_isi'] = float(stats['std_isi_ms'] / stats['mean_isi_ms']) if stats['mean_isi_ms'] > 0 else 0.0
    else:
        stats['mean_rate_hz'] = 0.0
        stats['mean_isi_ms'] = 0.0
        stats['std_isi_ms'] = 0.0
        stats['cv_isi'] = 0.0

    return stats

def plot_spike_detection(firing_pattern: np.ndarray, fs: float,
                         spike_indices: np.ndarray, spike_times_s: np.ndarray,
                         threshold: float, channel_name: str = "Channel",
                         component_idx: Optional[int] = None):
    """
    Plot firing pattern, detected spikes, raster plot, and ISI histogram.
    Creates three separate subplots with independent x-axes for better visibility.
    
    Args:
        firing_pattern: Original firing pattern
        fs: Sampling rate
        spike_indices: Detected spike indices
        spike_times_s: Spike times in seconds
        threshold: Used threshold value
        channel_name: Channel name for plot title
        component_idx: Component index for multi-component data
    """
    t = np.arange(len(firing_pattern)) / fs
    stats = analyze_spike_statistics(spike_times_s)

    # Create figure with three separate subplots (no shared x-axis)
    fig, axs = plt.subplots(3, 1, figsize=(14, 11)) 
    
    # 1) Firing pattern + detected spikes
    axs[0].plot(t, firing_pattern, lw=1.0, label='Firing pattern', color='C0', alpha=0.8)
    axs[0].axhline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold:.4g}')
    
    if len(spike_indices) > 0:
        axs[0].plot(spike_times_s, firing_pattern[spike_indices], 'ro', 
                   markersize=6, label=f'Detected spikes (n={len(spike_indices)})')
    
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Firing Pattern Amplitude (a.u.)')
    
    title_str = f'Firing Pattern with Detected Spikes'
    if component_idx is not None:
        title_str += f' (Component {component_idx+1})'
    title_str += f'\nRate: {stats["mean_rate_hz"]:.2f} Hz, Spikes: {len(spike_indices)}'
    
    axs[0].set_title(title_str, fontsize=11, pad=10) 
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # Set reasonable x-axis limits for firing pattern
    if len(t) > 0:
        axs[0].set_xlim(t[0], t[-1])

    # 2) Raster plot (spike timing visualization)
    if len(spike_times_s) > 0:
        # Create raster plot with better visualization
        axs[1].eventplot([spike_times_s], colors='black', linewidths=2, 
                        linelengths=0.8, alpha=0.8)
        axs[1].set_xlim(spike_times_s[0] - 0.1, spike_times_s[-1] + 0.1)
        
        # Add some statistics to the raster plot
        if len(spike_times_s) >= 2:
            duration = spike_times_s[-1] - spike_times_s[0]
            axs[1].text(0.02, 0.5, f'Duration: {duration:.2f}s\nTotal spikes: {len(spike_times_s)}', 
                       transform=axs[1].transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    else:
        axs[1].text(0.5, 0.5, 'No spikes detected', ha='center', va='center', 
                   transform=axs[1].transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Spike Events')
    axs[1].set_title('Spike Raster Plot (Spike Timing)', fontsize=11, pad=8)  
    axs[1].set_yticks([])
    axs[1].grid(True, alpha=0.3, axis='x')

    # 3) ISI histogram and analysis
    if len(spike_times_s) >= 2:
        isi_ms = np.diff(spike_times_s) * 1000.0
        n_bins = min(30, max(8, len(isi_ms) // 2))
        
        # Create histogram
        counts, bins, patches = axs[2].hist(isi_ms, bins=n_bins, color='lightblue', 
                                           edgecolor='darkblue', alpha=0.7, density=False)
        
        # Calculate mean and median ISI
        mean_isi = np.mean(isi_ms)
        median_isi = np.median(isi_ms)
        
        # Draw mean and median lines
        mean_line = axs[2].axvline(mean_isi, color='red', linestyle='--', linewidth=2)
        median_line = axs[2].axvline(median_isi, color='orange', linestyle=':', linewidth=2)
        
        # Prepare legend labels and handles
        cv_isi = stats["cv_isi"]
        std_isi = stats["std_isi_ms"]
        legend_labels = [
            f'Mean ISI = {mean_isi:.1f} ms',
            f'Median ISI = {median_isi:.1f} ms',
            f'CV = {cv_isi:.3f}',
            f'Std = {std_isi:.1f} ms'
        ]
        legend_handles = [
            mean_line,
            median_line,
            # Use invisible lines for CV and Std legend entries
            plt.Line2D([0], [0], color='none'),
            plt.Line2D([0], [0], color='none')
        ]
        axs[2].legend(legend_handles, legend_labels, loc='upper right', fontsize=10, framealpha=0.9)
        
        axs[2].set_xlabel('Inter-Spike Interval (ms)')
        axs[2].set_ylabel('Count')
        axs[2].set_title('Inter-Spike Interval Distribution', fontsize=11, pad=8)  
        axs[2].grid(True, alpha=0.3)
        
        # Set reasonable x-axis limits
        if len(isi_ms) > 0:
            x_min = max(0, np.min(isi_ms) - 5)
            x_max = np.max(isi_ms) + 10
            axs[2].set_xlim(x_min, x_max)
    else:
        # Not enough spikes for ISI analysis
        axs[2].text(0.5, 0.5, 'Insufficient spikes for ISI analysis\n(Need ≥ 2 spikes)', 
                   ha='center', va='center', transform=axs[2].transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        axs[2].set_xlabel('Inter-Spike Interval (ms)')
        axs[2].set_ylabel('Count')
        axs[2].set_title('Inter-Spike Interval Distribution', fontsize=11, pad=8)
        axs[2].grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.94])  
    
    fig.suptitle(f'Spike Detection Analysis - {channel_name}', 
                fontsize=14, fontweight='bold', y=0.97)
    
    return fig

def plot_kernels_and_spikes(kernels: np.ndarray,
                            spike_results: Dict,
                            fs: float,
                            channel_name: str = "Channel"):
    """
    Plot kernels (left) and spike trains (right) for each component.
    kernels: shape (kernel_len, n_components)
    spike_results: dict of {comp_name: comp_results} from process_single_dataset_spikes
    fs: sampling rate
    """
    n_components = kernels.shape[1]
    kernel_len = kernels.shape[0]
    kernel_t = np.linspace(-kernel_len//2/fs, kernel_len//2/fs, kernel_len)

    fig, axs = plt.subplots(n_components, 2, figsize=(10, 3.5 * n_components))
    if n_components == 1:
        axs = np.array([axs])

    for i, (comp_name, comp_data) in enumerate(spike_results['components'].items()):
        # Left: kernel
        axs[i, 0].plot(kernel_t, kernels[:, i], color='C0')
        axs[i, 0].set_title(f"{channel_name} — Kernel {i+1}")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_ylabel("Amplitude (a.u.)")
        axs[i, 0].grid(True, alpha=0.3)

        # Right: spikes (binary vector)
        spike_indices = comp_data['spike_indices']
        n_samples = len(comp_data['firing_pattern'])
        t = np.arange(n_samples) / fs
        spike_vec = np.zeros(n_samples)
        spike_vec[spike_indices] = 1

        axs[i, 1].stem(t, spike_vec, markerfmt=' ', basefmt=' ')
        axs[i, 1].set_title(f"{channel_name} — Spikes {i+1}")
        axs[i, 1].set_xlabel("Time (s)")
        axs[i, 1].set_ylabel("Spike (0/1)")
        axs[i, 1].set_ylim(-0.1, 1.1)
        axs[i, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{channel_name}: Kernels and Spikes", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def load_pickle_data(dataset_path: str) -> Dict:
    """
    Load pickle deconvolution results from EMG_to_MUAP.py output.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary containing loaded pickle data organized by mode and channel
    """
    pickle_data = {}
    
    for mode in ['single', 'multi']:
        mode_dir = os.path.join(dataset_path, 'multi-kernel-deconv', mode)
        if not os.path.exists(mode_dir):
            print(f"  Directory not found: {mode_dir}")
            continue
        
        # Find pickle file
        pattern = os.path.join(mode_dir, f'results_{mode}_*.pkl')
        pkl_files = glob.glob(pattern)
        if not pkl_files:
            print(f"  No pickle files found in {mode_dir}")
            continue
        
        pkl_file = sorted(pkl_files)[-1]  # Latest file
        print(f"  Loading {mode} mode from: {os.path.basename(pkl_file)}")
        
        try:
            with open(pkl_file, 'rb') as f:
                deconv_results = pickle.load(f)
            
            # deconv_results is a list of dicts, one per channel
            # Each dict contains: sigmas, mode, K, X, recon_total, recon_components, 
            #                     CWF, CWF_comp, conv_history, alpha, iterations, residual
            
            if not deconv_results:
                print(f"    {mode} mode: no results in pickle file")
                continue
            
            mode_data = {}
            channel_labels = ['LGA', 'RGA', 'LTA', 'RTA', 'LBF', 'RBF', 'LRF', 'RRF']
            
            for ch_idx, ch_result in enumerate(deconv_results):
                if ch_idx >= len(channel_labels):
                    break
                    
                ch_label = channel_labels[ch_idx]
                ch_data = {
                    'kernel': ch_result['K'],                    # shape (kernel_len, n_kernels)
                    'firing_pattern': ch_result['X'],            # shape (N, n_kernels)
                    'CWF': ch_result['CWF'],                     # shape (N,)
                    'CWF_comp': ch_result['CWF_comp'],           # shape (N, n_kernels)
                    'recon_total': ch_result['recon_total'],     # shape (N,)
                    'recon_components': ch_result['recon_components'],  # shape (N, n_kernels)
                    'sigmas': ch_result['sigmas'],
                    'alpha': ch_result['alpha'],
                    'iterations': ch_result['iterations'],
                    'residual': ch_result['residual']
                }
                
                mode_data[ch_label] = ch_data
                print(f"    Found {ch_label}: firing_pattern shape {ch_result['X'].shape}")
            
            if mode_data:
                pickle_data[mode] = mode_data
                print(f"    {mode} mode: loaded data for {len(mode_data)} channels")
        
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return pickle_data

def process_single_dataset_spikes(dataset_path: str, 
                                 fs: float = 1000.0,
                                 spike_params: Optional[Dict] = None,
                                 save_results: bool = True,
                                 plot_results: bool = True,
                                 exclude_time_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Process firing patterns from a single dataset to extract spikes.
    Each column of firing pattern array is processed separately.
    
    Args:
        dataset_path: Path to individual dataset (e.g., path2_01)
        fs: Sampling rate (Hz)
        spike_params: Parameters for spike detection
        save_results: Whether to save results
        plot_results: Whether to create and save plots
        exclude_time_range: Optional (start_time, end_time) in seconds to exclude spikes
                          e.g., (0, 5) to exclude baseline period
        
    Returns:
        Dictionary with spike data for each channel, mode, and component
    """
    if spike_params is None:
        spike_params = {
            'use_percentile': 97,
            'k_mad': 3.0,
            'smooth_sigma_ms': 0.0,
            'min_isi_ms': 30.0,
            'min_prominence': None
        }
    
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing dataset: {dataset_name}")
    
    # Load pickle data
    pickle_data = load_pickle_data(dataset_path)
    if not pickle_data:
        print(f"No pickle data found in {dataset_path}")
        return {}
    
    # Create output directory
    output_dir = os.path.join(dataset_path, 'firing_pattern_to_spikes')
    if save_results or plot_results:
        os.makedirs(output_dir, exist_ok=True)
    
    spike_results = {}
    channel_labels = ['LGA', 'RGA', 'LTA', 'RTA', 'LBF', 'RBF', 'LRF', 'RRF']
    
    for mode in ['single', 'multi']:
        if mode not in pickle_data:
            continue
        
        mode_results = {}
        mode_data = pickle_data[mode]
        print(f"\nProcessing {mode.upper()} mode:")
        
        for ch_label in channel_labels:
            if ch_label not in mode_data or 'firing_pattern' not in mode_data[ch_label]:
                continue
            
            firing_pattern = mode_data[ch_label]['firing_pattern']
            print(f"  {ch_label}: Shape {firing_pattern.shape}")
            
            # Handle multi-dimensional firing patterns (process each column separately)
            if firing_pattern.ndim == 1:
                # Single component
                firing_pattern = firing_pattern.reshape(-1, 1)
            
            n_components = firing_pattern.shape[1] if firing_pattern.ndim > 1 else 1
            ch_results = {'components': {}}
            
            for comp_idx in range(n_components):
                # Extract single column/component
                if firing_pattern.ndim > 1:
                    component_data = firing_pattern[:, comp_idx]
                else:
                    component_data = firing_pattern
                
                # Skip if component is all zeros or very small values
                if np.max(np.abs(component_data)) < 1e-8:
                    print(f"    Component {comp_idx+1}: Skipped (near-zero values)")
                    continue
                
                # Extract spikes from this component
                spike_indices, spike_times, binary_vec, threshold, suppressed_indices = convert_firing_to_spikes(
                    component_data, fs, exclude_time_range=exclude_time_range, **spike_params
                )

                # Analyze statistics
                stats = analyze_spike_statistics(spike_times)
                
                # Store results for this component
                comp_results = {
                    'firing_pattern': component_data,
                    'spike_indices': spike_indices,
                    'spike_times_s': spike_times,
                    'binary_vector': binary_vec,
                    'threshold_used': threshold,
                    'statistics': stats,
                    'fs': fs,
                    'component_index': comp_idx
                }
                
                ch_results['components'][f'comp_{comp_idx+1}'] = comp_results
                
                print(f"    Component {comp_idx+1}: {len(spike_indices)} spikes detected "
                      f"(Rate: {stats['mean_rate_hz']:.1f} Hz, "
                      f"Mean ISI: {stats['mean_isi_ms']:.1f} ms)")
                
                # Create and save plot if requested
                if plot_results and len(spike_indices) > 0:
                    fig = plot_spike_detection(
                        component_data, fs, spike_indices, spike_times, 
                        threshold, f"{ch_label}", comp_idx
                    )
                    
                    plot_filename = f'spikes_{mode}_{ch_label}_comp{comp_idx+1}_{dataset_name}.png'
                    plot_path = os.path.join(output_dir, plot_filename)
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"      Plot saved: {plot_filename}")
            
            # Store channel results
            if ch_results['components']:
                ch_results['n_components'] = len(ch_results['components'])
                ch_results['original_shape'] = firing_pattern.shape
                mode_results[ch_label] = ch_results

            # Plot kernels and spikes if available
            if plot_results and 'kernel' in mode_data[ch_label]:
                kernels = mode_data[ch_label]['kernel']  # shape (kernel_len, n_components)
                fig = plot_kernels_and_spikes(kernels, ch_results, fs, channel_name=ch_label)
                plot_filename = f'kernels_spikes_{mode}_{ch_label}_{dataset_name}.png'
                plot_path = os.path.join(output_dir, plot_filename)
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Kernel+Spikes plot saved: {plot_filename}")

        if mode_results:
            spike_results[mode] = mode_results
    
    # Save complete results
    if save_results and spike_results:
        # Save as pickle file
        results_path = os.path.join(output_dir, f'spike_results_{dataset_name}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(spike_results, f)
        print(f"Results saved: {results_path}")
        
        # Save summary CSV
        summary_data = []
        for mode, mode_data in spike_results.items():
            for ch_label, ch_data in mode_data.items():
                for comp_name, comp_data in ch_data['components'].items():
                    stats = comp_data['statistics']
                    summary_data.append({
                        'dataset': dataset_name,
                        'mode': mode,
                        'channel': ch_label,
                        'component': comp_name,
                        'component_idx': comp_data['component_index'],
                        'spike_count': stats['spike_count'],
                        'mean_rate_hz': stats['mean_rate_hz'],
                        'mean_isi_ms': stats['mean_isi_ms'],
                        'std_isi_ms': stats['std_isi_ms'],
                        'cv_isi': stats['cv_isi'],
                        'threshold_used': comp_data['threshold_used'],
                        'fs': comp_data['fs']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, f'spike_summary_{dataset_name}.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved: {summary_path}")
    
    return spike_results

def process_all_datasets_spikes(base_path: str, 
                               prefix: str = 'path2_',
                               fs: float = 1000.0,
                               spike_params: Optional[Dict] = None,
                               exclude_time_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Process all datasets in base_path to extract spikes from firing patterns.
    
    Args:
        base_path: Base directory containing datasets
        prefix: Dataset folder prefix
        fs: Sampling rate (Hz)
        spike_params: Parameters for spike detection
        exclude_time_range: Optional (start_time, end_time) in seconds to exclude spikes
        
    Returns:
        Dictionary with results for all datasets
    """
    datasets = sorted([d for d in glob.glob(os.path.join(base_path, f'{prefix}*')) 
                      if os.path.isdir(d)])
    
    if not datasets:
        print(f"No {prefix} datasets found in {base_path}")
        return {}
    
    print(f"Found {len(datasets)} datasets to process")
    all_results = {}
    
    for i, dataset_path in enumerate(datasets, 1):
        dataset_name = os.path.basename(dataset_path)
        print(f"\n--- Processing {i}/{len(datasets)}: {dataset_name} ---")
        
        try:
            results = process_single_dataset_spikes(
                dataset_path, fs=fs, spike_params=spike_params,
                save_results=True, plot_results=True,
                exclude_time_range=exclude_time_range
            )
            
            if results:
                all_results[dataset_name] = results
                print(f"✓ Successfully processed {dataset_name}")
            else:
                print(f"✗ No results for {dataset_name}")
        
        except Exception as e:
            print(f"✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save master summary
    if all_results:
        master_summary = []
        for dataset_name, dataset_results in all_results.items():
            for mode, mode_data in dataset_results.items():
                for ch_label, ch_data in mode_data.items():
                    n_components = ch_data['n_components']
                    total_spikes = sum(comp['statistics']['spike_count'] 
                                     for comp in ch_data['components'].values())
                    avg_rate = np.mean([comp['statistics']['mean_rate_hz'] 
                                       for comp in ch_data['components'].values()])
                    
                    master_summary.append({
                        'dataset': dataset_name,
                        'mode': mode,
                        'channel': ch_label,
                        'n_components': n_components,
                        'total_spikes': total_spikes,
                        'avg_rate_hz': avg_rate,
                        'original_shape': str(ch_data['original_shape'])
                    })
        
        if master_summary:
            master_df = pd.DataFrame(master_summary)
            master_path = os.path.join(base_path, 'spike_detection_master_summary.csv')
            master_df.to_csv(master_path, index=False)
            print(f"\nMaster summary saved: {master_path}")
    
    return all_results

# ---------------------- Real Data Processing ----------------------
if __name__ == "__main__":
    # Configuration matching your setup
    base_path = '/home/ntk/exo/ExoPulse/simulation/mocap_EMG_EEG_data/20241225/'
    fs = 1000.0  # Sampling rate in Hz

    # Test with path2_02 using percentile threshold
    test_dataset = os.path.join(base_path, 'path2_02')
    if os.path.exists(test_dataset):
        print("\n=== Percentile Threshold Detection for path2_02 ===")
        print("Excluding baseline period: 0-5 seconds")
        spike_params = {
            'threshold': None,
            'use_percentile': 97,  # 97th percentile
            'k_mad': 3.0,
            'smooth_sigma_ms': 0.0,
            'min_isi_ms': 30.0,
            'min_prominence': None
        }
        results = process_single_dataset_spikes(
            test_dataset,
            fs=fs,
            spike_params=spike_params,
            save_results=True,
            plot_results=True,
            exclude_time_range=(0, 8)  # Exclude baseline period (0-8 seconds)
        )
        total_spikes = 0
        for mode, mode_data in results.items():
            for ch_label, ch_data in mode_data.items():
                for comp_data in ch_data['components'].values():
                    total_spikes += comp_data['statistics']['spike_count']
        print(f"Total spikes detected: {total_spikes}")
    else:
        print(f"Test dataset not found: {test_dataset}")
# %%