# %%
"""
Spike-Triggered Analysis Module for EEG-EMG Correlation

Performs spike-triggered average (STA) and pairwise phase consistency (PPC) analysis
between EMG-derived spikes and EEG signals across movement states and frequency bands.

Key Features:
- STA: Average EEG response locked to spike times (uses frequency-specific windows)
- PPC: Phase synchronization by sampling instantaneous phase at spike times (no window)
- Frequency-specific extraction windows for STA (matched to oscillation periods)
- Uniform visualization window (±300ms) across all bands for comparison
- Works with relative time coordinates (0-based) from segmented data

Frequency-Specific Analysis Windows:
Extraction Windows (capturing appropriate oscillation cycles):
  - Delta (1-4 Hz):     ±500ms  (1-2 cycles)
  - Theta (4-8 Hz):     ±500ms  (2-4 cycles)
  - Mu/Alpha (8-12 Hz): ±300ms  (2.4-3.6 cycles)
  - Beta (12-30 Hz):    ±150ms  (1.8-4.5 cycles)
  - Low Gamma (30-46 Hz): ±75ms (2.25-3.45 cycles)

Display Window: ±300ms (uniform across all bands for visual comparison)

Target EEG Channels:
- Motor: Cz, C3, C4, FCz, FC3, FC4, CPz, CP3, CP4 (use mu band)
- Visual: O1, O2, Oz (use alpha band)

Movement States:
- sit_to_stand, stand_to_walk, steady_gait, walk_to_stop
"""

import os
import glob
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import circmean, circstd
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d


class SpikeTriggeredAnalyzer:
    """Spike-triggered analysis for EEG-EMG correlation: STA and PPC measures
    
    Works with RELATIVE time coordinates (0-based) from neural_data_movement_segmentor.
    All time arrays within each movement segment start from 0 and share the same timebase.
    """
    
    def __init__(self, eeg_fs: float = 500.0, emg_fs: float = 1000.0):
        self.eeg_fs = eeg_fs
        self.emg_fs = emg_fs
        
        self.motor_freq_bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12),
            'beta': (12, 30), 'low_gamma': (30, 46)
        }
        
        self.visual_freq_bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12),
            'beta': (12, 30), 'low_gamma': (30, 46)
        }
        
        self.motor_channels = ['Cz', 'C3', 'C4', 'FCz', 'FC3', 'FC4', 'CPz', 'CP3', 'CP4']
        self.visual_channels = ['O1', 'O2', 'Oz']
        self.target_channels = self.motor_channels + self.visual_channels
        self.channel_groups = {'motor': self.motor_channels, 'visual': self.visual_channels}
        
        self.movement_states = ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']
        
        self.sta_windows = {
            'delta': 0.5, 'theta': 0.5, 'alpha': 0.3,
            'beta': 0.15, 'low_gamma': 0.075
        }
        self.display_window = 0.3
        
        self.motor_band_order = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']
        self.visual_band_order = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']
    
    def _get_freq_bands_for_channel(self, channel_name: str) -> Dict[str, Tuple[float, float]]:
        return self.visual_freq_bands if channel_name in self.visual_channels else self.motor_freq_bands
    
    def _get_band_order_for_channel(self, channel_name: str) -> List[str]:
        return self.visual_band_order if channel_name in self.visual_channels else self.motor_band_order
    
    def _verify_relative_time_coordinates(self, time_array: np.ndarray, spike_times: np.ndarray, 
                                         context: str = "") -> bool:
        """Verify that time coordinates are relative (start from ~0)
        
        Args:
            time_array: Time vector to check
            spike_times: Spike times array to check
            context: Context string for logging
            
        Returns:
            True if coordinates appear to be relative, False otherwise
        """
        if len(time_array) == 0:
            return True
        
        time_start = time_array[0]
        time_end = time_array[-1]
        
        # Check if time starts near 0 (within 0.01s tolerance)
        if abs(time_start) > 0.01:
            print(f"  Warning [{context}]: Time does not start from 0 (starts at {time_start:.3f}s)")
            print(f"    This may indicate absolute time coordinates instead of relative coordinates.")
            return False
        
        # Check if spike times are within time range
        if len(spike_times) > 0:
            spikes_in_range = np.sum((spike_times >= time_start) & (spike_times <= time_end))
            spikes_ratio = spikes_in_range / len(spike_times)
            
            if spikes_ratio < 0.5:
                print(f"  Warning [{context}]: Only {spikes_ratio*100:.1f}% of spikes are within time range")
                print(f"    Time range: [{time_start:.3f}, {time_end:.3f}]s")
                print(f"    Spike range: [{spike_times.min():.3f}, {spike_times.max():.3f}]s")
                return False
        
        return True
        
    def reorganize_eeg_data(self, segments_data: Dict) -> Dict:
        """Reorganize EEG from [eeg_file][state] to [state][channel] structure"""
        reorganized_eeg = {state: {} for state in self.movement_states}
        eeg_segments = segments_data.get('eeg_segments', {})
        
        for eeg_filename, file_segments in eeg_segments.items():
            for state in self.movement_states:
                if state in file_segments and file_segments[state] is not None:
                    segment = file_segments[state]
                    channels = segment.get('channels', [])
                    data = segment['data']
                    time = segment['time']
                    
                    for ch_idx, channel_name in enumerate(channels):
                        if channel_name in self.target_channels:
                            if data.ndim > 1 and ch_idx < data.shape[1]:
                                channel_data = data[:, ch_idx]
                            elif data.ndim == 1 and ch_idx == 0:
                                channel_data = data
                            else:
                                continue
                            
                            reorganized_eeg[state][channel_name] = {
                                'data': channel_data, 'time': time,
                                'fs': segment.get('fs', self.eeg_fs),
                                'source_file': eeg_filename,
                                'duration': segment.get('duration', 0),
                                'n_samples': len(channel_data),
                                'time_format': segment.get('time_format', 'relative'),
                                'absolute_start': segment.get('absolute_start', None),
                                'absolute_end': segment.get('absolute_end', None),
                                'data_start_time': segment.get('data_start_time', time[0])
                            }
        
        return reorganized_eeg
    
    def reorganize_spike_data(self, segments_data: Dict) -> Dict:
        """Reorganize spike data to {mode: {state: {comp_key: spike_data}}}"""
        reorganized_spikes = {}
        spike_segments = segments_data.get('spike_segments', {})
        
        for mode in ['single', 'multi']:
            if mode not in spike_segments:
                continue
            
            reorganized_spikes[mode] = {state: {} for state in self.movement_states}
            mode_data = spike_segments[mode]
            
            for ch_label, ch_segments in mode_data.items():
                for comp_name, comp_segments in ch_segments.items():
                    comp_key = f"{ch_label}_{comp_name}"
                    for state in self.movement_states:
                        if state in comp_segments and comp_segments[state] is not None:
                            segment = comp_segments[state]
                            reorganized_spikes[mode][state][comp_key] = {
                                'spike_times': segment['spike_times'],
                                'spike_indices': segment['spike_indices'],
                                'binary_train': segment['binary_train'],
                                'time': segment['time'],
                                'n_spikes': segment['n_spikes'],
                                'fs': segment.get('fs', self.emg_fs),
                                'duration': segment.get('duration', 0),
                                'channel': ch_label, 'component': comp_name,
                                'time_format': segment.get('time_format', 'relative'),
                                'absolute_start': segment.get('absolute_start', None),
                                'absolute_end': segment.get('absolute_end', None)
                            }
        
        return reorganized_spikes
    
    def bandpass_filter_eeg(self, eeg_data: np.ndarray, freq_band: Tuple[float, float], 
                       fs: float, filter_order: int = 4) -> np.ndarray:
        """Butterworth bandpass filter for EEG"""
        low_freq, high_freq = freq_band
        nyquist = fs / 2.0
        
        if high_freq >= nyquist * 0.99:
            high_freq = nyquist * 0.95
        if low_freq <= 0:
            low_freq = 0.5
        
        low_norm = np.clip(low_freq / nyquist, 0.001, 0.999)
        high_norm = np.clip(high_freq / nyquist, 0.001, 0.999)
        
        if low_norm >= high_norm:
            raise ValueError(f"Invalid frequency band: {low_freq}-{high_freq} Hz")
        
        b, a = butter(filter_order, [low_norm, high_norm], btype='bandpass')
        
        if eeg_data.ndim == 1:
            pad_length = max(1, min(len(eeg_data) // 2 - 1, 100))
            return filtfilt(b, a, eeg_data, padlen=pad_length)
        else:
            return np.array([
                filtfilt(b, a, eeg_data[:, ch], 
                        padlen=max(1, min(eeg_data.shape[0] // 2 - 1, 100)))
                for ch in range(eeg_data.shape[1])
            ]).T

    def extract_spike_triggered_segments(self, eeg_signal: np.ndarray, spike_times: np.ndarray,
                                       eeg_time: np.ndarray, window_size: float, 
                                       display_window: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract EEG segments around spike times (relative coordinates)
        
        With relative time coordinates (0-based), time alignment is automatic:
        - spike_times and eeg_time share the same timebase (both start from 0)
        - No coordinate transformation needed
        - Simple array indexing using np.argmin()
        
        Args:
            eeg_signal: EEG signal array (1D)
            spike_times: Spike occurrence times (relative coordinates, 0-based)
            eeg_time: EEG time vector (relative coordinates, 0-based)
            window_size: Extraction window size in seconds (frequency-specific)
            display_window: Display window size in seconds (uniform across bands)
            
        Returns:
            (segments, relative_time, display_time)
        """
        if len(spike_times) == 0:
            return np.array([]), np.array([]), np.array([])
        
        window_samples = int(window_size * self.eeg_fs)
        total_samples = 2 * window_samples + 1
        relative_time = np.linspace(-window_size, window_size, total_samples)
        display_time = relative_time if display_window is None else np.linspace(-display_window, display_window, total_samples)
        
        segments = []
        for spike_time in spike_times:
            spike_idx = np.argmin(np.abs(eeg_time - spike_time))
            start_idx = spike_idx - window_samples
            end_idx = spike_idx + window_samples + 1
            
            if start_idx >= 0 and end_idx <= len(eeg_signal):
                segment = eeg_signal[start_idx:end_idx]
                if len(segment) == total_samples:
                    segments.append(segment)
        
        segments = np.array(segments) if segments else np.array([]).reshape(0, total_samples)
        return segments, relative_time, display_time

    def compute_spike_triggered_average(self, eeg_segments: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute STA: mean, std, sem, n_spikes"""
        if eeg_segments.size == 0:
            return {'mean': np.array([]), 'std': np.array([]), 'sem': np.array([]), 'n_spikes': 0}
        
        n_spikes = eeg_segments.shape[0]
        sta_mean = np.mean(eeg_segments, axis=0)
        sta_std = np.std(eeg_segments, axis=0)
        sta_sem = sta_std / np.sqrt(n_spikes) if n_spikes > 1 else np.zeros_like(sta_std)
        
        return {'mean': sta_mean, 'std': sta_std, 'sem': sta_sem, 'n_spikes': n_spikes}

    def compute_pairwise_phase_consistency(self, instantaneous_phase: np.ndarray, 
                                           spike_times: np.ndarray,
                                           eeg_time: np.ndarray) -> Dict[str, float]:
        """Compute PPC and phase shift by sampling instantaneous phase at spike times
        
        Args:
            instantaneous_phase: Full EEG instantaneous phase array
            spike_times: Array of spike occurrence times
            eeg_time: EEG time axis
            
        Returns:
            Dictionary with PPC value, phase shift (mean phase angle), and statistics
            - ppc: Pairwise phase consistency (0-1, measures phase synchronization)
            - phase_shift: Mean phase angle in radians (-π to π, preferred phase relationship)
            - phase_shift_deg: Mean phase angle in degrees (-180 to 180)
            - phase_std: Circular standard deviation of phase (radians)
            - phase_std_deg: Circular standard deviation of phase (degrees)
        """
        if len(spike_times) == 0:
            return {
                'ppc': 0.0, 'n_pairs': 0, 'n_spikes': 0,
                'phase_shift': 0.0, 'phase_shift_deg': 0.0,
                'phase_std': 0.0, 'phase_std_deg': 0.0
            }
        
        # Sample phase at each spike time
        spike_phase_values = []
        for spike_time in spike_times:
            spike_idx = np.argmin(np.abs(eeg_time - spike_time))
            if 0 <= spike_idx < len(instantaneous_phase):
                spike_phase_values.append(instantaneous_phase[spike_idx])
        
        spike_phase_values = np.array(spike_phase_values)
        N = len(spike_phase_values)
        
        if N < 2:
            # Calculate phase shift even with single spike
            mean_phase = circmean(spike_phase_values, high=np.pi, low=-np.pi) if N == 1 else 0.0
            return {
                'ppc': 0.0, 'n_pairs': 0, 'n_spikes': N,
                'phase_shift': float(mean_phase),
                'phase_shift_deg': float(np.rad2deg(mean_phase)),
                'phase_std': 0.0,
                'phase_std_deg': 0.0
            }
        
        # PPC formula (Vinck et al., 2010)
        z = np.sum(np.exp(1j * spike_phase_values))
        ppc = (np.abs(z)**2 - N) / (N * (N - 1))
        
        # Phase shift: mean phase angle (circular mean)
        mean_phase = circmean(spike_phase_values, high=np.pi, low=-np.pi)
        phase_std = circstd(spike_phase_values, high=np.pi, low=-np.pi)
        
        return {
            'ppc': float(np.clip(ppc, 0, 1)),
            'n_pairs': N * (N - 1) // 2,
            'n_spikes': N,
            'phase_shift': float(mean_phase),
            'phase_shift_deg': float(np.rad2deg(mean_phase)),
            'phase_std': float(phase_std),
            'phase_std_deg': float(np.rad2deg(phase_std))
        }
    
    def analyze_channel_spike_correlation(self, eeg_data: np.ndarray, eeg_time: np.ndarray,
                                    spike_times: np.ndarray, channel_name: str,
                                    movement_state: str) -> Dict[str, Any]:
        """Spike-triggered analysis (STA and PPC) for single EEG channel with relative time coordinates
        
        STA and PPC are computed independently:
        - STA: Requires window extraction, may fail for boundary spikes
        - PPC: Direct phase sampling at spike times, always computed if spikes exist
        
        Time Coordinate System:
        - All time arrays (eeg_time, spike_times) use RELATIVE coordinates starting from 0
        - This is enforced by neural_data_movement_segmentor.py
        - Verification check ensures coordinate consistency
        """
        freq_bands = self._get_freq_bands_for_channel(channel_name)
        channel_type = 'Visual' if channel_name in self.visual_channels else 'Motor'
        
        # Verify relative time coordinates
        context = f"{channel_name}_{movement_state}"
        if not self._verify_relative_time_coordinates(eeg_time, spike_times, context):
            print(f"  Proceeding with analysis despite coordinate warnings...")
        
        results = {
            'channel': channel_name,
            'channel_type': channel_type,
            'movement_state': movement_state,
            'n_spikes': len(spike_times),
            'bands': {}
        }
        
        for band_name, freq_range in freq_bands.items():
            try:
                extraction_window = self.sta_windows.get(band_name, 0.5)
                filtered_eeg = self.bandpass_filter_eeg(eeg_data, freq_range, self.eeg_fs)
                
                # === STA Path: Extract segments (may fail for boundary spikes) ===
                eeg_segments, relative_time, display_time = self.extract_spike_triggered_segments(
                    filtered_eeg, spike_times, eeg_time, extraction_window, self.display_window
                )
                
                # Compute STA only if valid segments exist
                sta_results = None
                if len(eeg_segments) > 0:
                    sta_results = self.compute_spike_triggered_average(eeg_segments)
                
                # === PPC Path: Direct phase sampling (independent of window) ===
                analytic_signal = hilbert(filtered_eeg)
                instantaneous_phase = np.angle(analytic_signal)
                ppc_results = self.compute_pairwise_phase_consistency(
                    instantaneous_phase, spike_times, eeg_time
                )
                
                # Store results (even if STA is None, PPC is always computed)
                results['bands'][band_name] = {
                    'sta': sta_results,  # May be None if no valid segments
                    'ppc': ppc_results,  # Always computed (uses all spikes)
                    'n_valid_segments': len(eeg_segments),  # Number of spikes used for STA
                    'n_ppc_spikes': ppc_results.get('n_spikes', 0),  # Number of spikes used for PPC
                    'relative_time': relative_time,
                    'display_time': display_time,
                    'extraction_window': extraction_window,
                    'display_window': self.display_window,
                    'freq_range': freq_range
                }
            
            except Exception as e:
                print(f"Error in {band_name} ({channel_name}, {movement_state}): {e}")
                continue
        
        return results if results['bands'] else None
    
    def analyze_dataset_correlations(self, segments_data: Dict) -> Dict[str, Any]:
        """Analyze spike-EEG correlations across all movement states using relative time coordinates"""
        dataset_name = segments_data.get('dataset_name', 'unknown')
        print(f"\nAnalyzing dataset: {dataset_name}")
        
        results = {
            'dataset_name': dataset_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'movement_states': {},
            'summary_statistics': {}
        }
        
        reorganized_eeg = self.reorganize_eeg_data(segments_data)
        reorganized_spikes = self.reorganize_spike_data(segments_data)
        available_states = segments_data.get('available_states', [])
        
        for state in available_states:
            if state not in self.movement_states:
                continue
            
            results['movement_states'][state] = {}
            state_eeg = reorganized_eeg.get(state, {})
            
            if not state_eeg:
                continue
            
            for channel in self.target_channels:
                if channel not in state_eeg:
                    continue
                
                eeg_info = state_eeg[channel]
                eeg_data = eeg_info['data']
                eeg_time = eeg_info['time']
                channel_spike_results = {}
                
                for mode in ['single', 'multi']:
                    if mode not in reorganized_spikes:
                        continue
                    
                    mode_spikes = reorganized_spikes[mode].get(state, {})
                    
                    for comp_name, comp_data in mode_spikes.items():
                        if 'spike_times' not in comp_data or len(comp_data['spike_times']) < 2:
                            continue
                        
                        analysis_key = f"{mode}_{comp_name}"
                        spike_times = np.asarray(comp_data['spike_times'], dtype=float)
                        
                        n_in_range = np.sum((spike_times >= eeg_time[0]) & (spike_times <= eeg_time[-1]))
                        if n_in_range == 0:
                            continue
                        
                        correlation_results = self.analyze_channel_spike_correlation(
                            eeg_data, eeg_time, spike_times, channel, state
                        )
                        
                        if correlation_results is not None:
                            channel_spike_results[analysis_key] = correlation_results
                
                if channel_spike_results:
                    results['movement_states'][state][channel] = channel_spike_results
        
        results['summary_statistics'] = self.compute_summary_statistics(results)
        print(f"Analysis complete: {dataset_name}\n")
        
        return results
    
    def compute_summary_statistics(self, analysis_results: Dict) -> Dict[str, Any]:
        """Aggregate STA and PPC statistics across all analyses"""
        all_bands = set()
        movement_states = analysis_results.get('movement_states', {})
        
        for state_data in movement_states.values():
            for channel_data in state_data.values():
                for analysis_data in channel_data.values():
                    all_bands.update(analysis_data.get('bands', {}).keys())
        
        summary = {
            'total_analyses': 0,
            'channels_analyzed': set(),
            'states_analyzed': set(),
            'frequency_bands': sorted(list(all_bands)),
            'ppc_statistics': {band: [] for band in all_bands},
            'sta_statistics': {band: [] for band in all_bands}
        }
        
        for state_name, state_data in movement_states.items():
            summary['states_analyzed'].add(state_name)
            for channel_name, channel_data in state_data.items():
                summary['channels_analyzed'].add(channel_name)
                for analysis_key, analysis_data in channel_data.items():
                    summary['total_analyses'] += 1
                    bands = analysis_data.get('bands', {})
                    for band_name, band_data in bands.items():
                        ppc_val = band_data.get('ppc', {}).get('ppc', 0.0)
                        if ppc_val > 0:
                            summary['ppc_statistics'][band_name].append(ppc_val)
                        sta_stats = band_data.get('sta', {})
                        if sta_stats is not None and 'mean' in sta_stats and len(sta_stats['mean']) > 0:
                            summary['sta_statistics'][band_name].append(np.max(np.abs(sta_stats['mean'])))
        
        summary['channels_analyzed'] = sorted(list(summary['channels_analyzed']))
        summary['states_analyzed'] = sorted(list(summary['states_analyzed']))
        
        for band in all_bands:
            for measure in ['ppc_statistics', 'sta_statistics']:
                values = summary[measure][band]
                summary[measure][band] = {
                    'values': values,
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'median': float(np.median(values)) if values else 0.0,
                    'min': float(np.min(values)) if values else 0.0,
                    'max': float(np.max(values)) if values else 0.0,
                    'n_samples': len(values)
                }
        
        return summary
    
    def visualize_correlation_results(self, analysis_results: Dict, output_folder: str = None):
        """Generate PPC and STA visualizations organized by state, mode, and channel"""
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        print("\nGenerating visualizations...")
        self._plot_ppc_by_mode_channel(analysis_results, output_folder)
        self._plot_sta_by_mode_channel(analysis_results, output_folder)
        
        if output_folder:
            print(f"\n✓ All visualizations saved to: {output_folder}")

    def _plot_sta_by_mode_channel(self, analysis_results: Dict, output_folder: str = None):
        """Plot STA results by movement state, mode, and EMG channel"""
        sns.set_style('whitegrid')
        movement_states = analysis_results.get('movement_states', {})
        band_colors = {
            'delta': '#1f77b4', 'theta': '#ff7f0e', 'alpha': '#2ca02c',
            'beta': '#d62728', 'low_gamma': '#9467bd'
        }

        for state_name, state_data in movement_states.items():
            if not state_data:
                continue

            for eeg_channel_name, channel_data in state_data.items():
                channel_type = "Motor Control" if eeg_channel_name in self.motor_channels else (
                    "Visual Processing" if eeg_channel_name in self.visual_channels else "Other")

                components_by_mode_emg = {}
                for analysis_key, analysis_data in channel_data.items():
                    if '_' in analysis_key:
                        parts = analysis_key.split('_')
                        mode = parts[0]
                        emg_channel = parts[1] if len(parts) >= 3 else 'unknown'
                        comp_name = '_'.join(parts[2:]) if len(parts) >= 3 else analysis_key
                        mode_emg_key = f"{mode}_{emg_channel}"
                        if mode_emg_key not in components_by_mode_emg:
                            components_by_mode_emg[mode_emg_key] = {}
                        components_by_mode_emg[mode_emg_key][comp_name] = analysis_data

                for mode_emg_key, components in components_by_mode_emg.items():
                    if not components:
                        continue
                    parts = mode_emg_key.split('_')
                    mode = parts[0]
                    emg_channel = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'

                    for comp_name, analysis_data in components.items():
                        bands = analysis_data.get('bands', {})
                        if not bands:
                            continue
                        n_bands = len(bands)
                        if n_bands == 0:
                            continue

                        fig, axes = plt.subplots(n_bands, 1, figsize=(14, 4 * n_bands))
                        if n_bands == 1:
                            axes = [axes]
                        has_data = False
                        band_idx = 0

                        for band_name, band_data in bands.items():
                            if band_idx >= len(axes):
                                break
                            ax = axes[band_idx]
                            sta_stats = band_data.get('sta', {})
                            display_time = band_data.get('display_time', [])
                            relative_time = band_data.get('relative_time', [])
                            extraction_window = band_data.get('extraction_window', 0.5)
                            freq_range = band_data.get('freq_range', (0, 0))
                            time_axis = display_time if len(display_time) > 0 else relative_time

                            if sta_stats is not None and 'mean' in sta_stats and len(sta_stats['mean']) > 0:
                                mean = sta_stats['mean']
                                sem = sta_stats.get('sem', np.zeros_like(mean))
                                n_spikes = sta_stats.get('n_spikes', 0)
                                total_spikes = analysis_data.get('n_spikes', 0)
                                color = band_colors.get(band_name, '#000000')
                                ax.plot(time_axis, mean, '-', linewidth=2.5, color=color, alpha=0.8)
                                ax.fill_between(time_axis, mean - sem, mean + sem, color=color, alpha=0.15)
                                ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Spike time')
                                ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                                band_title = f'{band_name.title()} ({freq_range[0]}-{freq_range[1]} Hz)'
                                window_info = f'extraction: ±{extraction_window*1000:.0f}ms'
                                spike_info = f'spikes: {n_spikes}/{total_spikes}'
                                ax.set_title(f'{band_title} | {window_info} | {spike_info}', 
                                        fontsize=11, fontweight='bold', loc='left')
                                ax.set_xlabel('Time relative to spike (s)', fontsize=10)
                                ax.set_ylabel('EEG Amplitude (µV)', fontsize=10)
                                ax.grid(True, alpha=0.3, linestyle='--')
                                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                                has_data = True
                            band_idx += 1

                        if not has_data:
                            plt.close(fig)
                            continue

                        state_title = state_name.replace("_", " ").title()
                        mode_title = mode.title()
                        plt.suptitle(
                            f'Spike-Triggered Average - All Frequency Bands\n'
                            f'{state_title} | {mode_title} Mode | EMG: {emg_channel} | Component: {comp_name}\n'
                            f'EEG: {eeg_channel_name} ({channel_type})',
                            fontsize=13, fontweight='bold', y=0.998
                        )
                        plt.tight_layout()

                        if output_folder:
                            state_folder_name = state_name.replace("_", " ").title()
                            plot_save_folder = os.path.join(output_folder, eeg_channel_name, 
                                                        mode_title, 'STA', state_folder_name)
                            os.makedirs(plot_save_folder, exist_ok=True)
                            filename_base = f'sta_{emg_channel}_{comp_name}'
                            png_path = os.path.join(plot_save_folder, f'{filename_base}.png')
                            plt.savefig(png_path, dpi=300, bbox_inches='tight')
                            print(f"  Saved STA: {os.path.relpath(png_path, output_folder)}")
                        plt.close(fig)

    def _plot_ppc_by_mode_channel(self, analysis_results: Dict, output_folder: str = None):
        """PPC heatmap per state/EEG channel/mode: rows=frequency bands, columns=EMG components"""
        sns.set_style('whitegrid')
        movement_states = analysis_results.get('movement_states', {})

        for state_name, state_data in movement_states.items():
            if not state_data:
                continue

            for eeg_channel_name, channel_data in state_data.items():
                channel_type = "Motor Control" if eeg_channel_name in self.motor_channels else (
                    "Visual Processing" if eeg_channel_name in self.visual_channels else "Other")

                components_by_mode = {'single': {}, 'multi': {}}
                for analysis_key, analysis_data in channel_data.items():
                    if '_' in analysis_key:
                        parts = analysis_key.split('_')
                        mode = parts[0]
                        emg_channel = parts[1] if len(parts) >= 3 else 'unknown'
                        comp_name = '_'.join(parts[2:]) if len(parts) >= 3 else analysis_key
                        column_label = f"{emg_channel}-{comp_name}"
                        if mode in components_by_mode:
                            components_by_mode[mode][column_label] = analysis_data

                for mode, components in components_by_mode.items():
                    if not components:
                        continue

                    all_bands = set()
                    for analysis_data in components.values():
                        all_bands.update(analysis_data.get('bands', {}).keys())
                    band_order = self._get_band_order_for_channel(eeg_channel_name)
                    bands = [b for b in band_order if b in all_bands]
                    component_names = list(components.keys())
                    if len(component_names) == 0 or len(bands) == 0:
                        continue

                    ppc_df = pd.DataFrame(index=bands, columns=component_names, dtype=float)
                    nsp_df = pd.DataFrame(index=bands, columns=component_names, dtype=int)
                    phase_df = pd.DataFrame(index=bands, columns=component_names, dtype=float)
                    phase_std_df = pd.DataFrame(index=bands, columns=component_names, dtype=float)
                    freq_ranges = {}

                    for comp_label, analysis_data in components.items():
                        band_dict = analysis_data.get('bands', {})
                        for band_name in bands:
                            if band_name in band_dict:
                                ppc_val = band_dict[band_name].get('ppc', {}).get('ppc', np.nan)
                                n_spikes = int(band_dict[band_name].get('ppc', {}).get('n_spikes', 0))
                                phase_shift_deg = band_dict[band_name].get('ppc', {}).get('phase_shift_deg', np.nan)
                                phase_std_deg = band_dict[band_name].get('ppc', {}).get('phase_std_deg', np.nan)
                                freq_range = band_dict[band_name].get('freq_range', (0, 0))
                                ppc_df.at[band_name, comp_label] = ppc_val
                                nsp_df.at[band_name, comp_label] = n_spikes
                                phase_df.at[band_name, comp_label] = phase_shift_deg
                                phase_std_df.at[band_name, comp_label] = phase_std_deg
                                freq_ranges[band_name] = freq_range
                            else:
                                ppc_df.at[band_name, comp_label] = np.nan
                                nsp_df.at[band_name, comp_label] = 0
                                phase_df.at[band_name, comp_label] = np.nan
                                phase_std_df.at[band_name, comp_label] = np.nan

                    if ppc_df.isna().all().all():
                        continue

                    def sort_key(col_name):
                        parts = col_name.split('-')
                        return (parts[0] if len(parts) > 0 else '', parts[1] if len(parts) > 1 else '')
                    col_order = sorted(component_names, key=sort_key)
                    ppc_df = ppc_df[col_order]
                    nsp_df = nsp_df[col_order]
                    phase_df = phase_df[col_order]
                    phase_std_df = phase_std_df[col_order]

                    # Create annotations with PPC, phase shift ± std, and spike count
                    annot = ppc_df.copy().astype(object)
                    for r in ppc_df.index:
                        for c in ppc_df.columns:
                            val = ppc_df.at[r, c]
                            nsp = int(nsp_df.at[r, c])
                            phase = phase_df.at[r, c]
                            phase_std = phase_std_df.at[r, c]
                            if np.isnan(val):
                                annot.at[r, c] = ""
                            else:
                                # Format: PPC value, phase shift ± std in degrees, spike count
                                annot.at[r, c] = f"{val:.3f}\nφ={phase:.0f}°±{phase_std:.0f}°\n(n={nsp})"

                    # Adjust figure size for optimal readability: larger cells to prevent text overlap
                    figsize = (max(16, len(component_names) * 1.5), max(10, len(bands) * 2.2))
                    fig, ax = plt.subplots(figsize=figsize)
                    sns.heatmap(ppc_df.astype(float), annot=annot, fmt='', 
                               cmap='viridis', vmin=0.0, vmax=1.0, 
                               linewidths=0.8, linecolor='white',
                               cbar_kws={'label': 'PPC (0-1)', 'shrink': 0.8}, ax=ax,
                               annot_kws={'fontsize': 13, 'va': 'center', 'ha': 'center', 
                                         'linespacing': 1.3, 'color': 'white'})

                    state_title = state_name.replace("_", " ").title()
                    mode_title = mode.title()
                    ax.set_title(
                        f'PPC & Phase Shift Heatmap — All EMG Channels & Frequency Bands\n'
                        f'{state_title} | {mode_title} Mode | '
                        f'EEG: {eeg_channel_name} ({channel_type})\n'
                        f'(PPC value, φ=phase shift±std in degrees, n=spike count)',
                        fontsize=12, fontweight='bold'
                    )
                    ax.set_xlabel('EMG Channel - Spike Component', fontsize=11)
                    ax.set_ylabel('Frequency Band', fontsize=11)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                    y_labels = [f'{band_name.title()}\n({freq_ranges.get(band_name, (0, 0))[0]}-{freq_ranges.get(band_name, (0, 0))[1]} Hz)' 
                               for band_name in ppc_df.index]
                    ax.set_yticklabels(y_labels, fontsize=9)
                    
                    current_emg = None
                    for idx, col_name in enumerate(col_order):
                        emg_ch = col_name.split('-')[0]
                        if current_emg is not None and emg_ch != current_emg:
                            ax.axvline(idx, color='white', linewidth=2.5, alpha=0.8)
                        current_emg = emg_ch
                    plt.tight_layout()

                    if output_folder:
                        plot_save_folder = os.path.join(output_folder, eeg_channel_name, mode_title, 'PPC')
                        os.makedirs(plot_save_folder, exist_ok=True)
                        filename = f'ppc_heatmap_{state_name}_all_channels.png'
                        png_path = os.path.join(plot_save_folder, filename)
                        plt.savefig(png_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved PPC: {os.path.relpath(png_path, output_folder)}")
                    plt.close(fig)

    def save_analysis_results(self, analysis_results: Dict, output_folder: str):
        """Save analysis results to pickle (complete) and CSV (summary)"""
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = analysis_results.get('dataset_name', 'unknown')
        
        pickle_path = os.path.join(output_folder, f'correlation_analysis_{dataset_name}_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(analysis_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        summary = analysis_results.get('summary_statistics', {})
        summary_rows = []
        all_bands = summary.get('frequency_bands', [])
        
        for band_name in all_bands:
            for measure in ['ppc_statistics', 'sta_statistics']:
                stats = summary.get(measure, {}).get(band_name, {})
                if isinstance(stats, dict) and 'mean' in stats:
                    freq_range_str = "N/A"
                    for state_data in analysis_results.get('movement_states', {}).values():
                        for channel_data in state_data.values():
                            for analysis_data in channel_data.values():
                                if band_name in analysis_data.get('bands', {}):
                                    freq_range = analysis_data['bands'][band_name].get('freq_range', (0, 0))
                                    freq_range_str = f"{freq_range[0]}-{freq_range[1]}"
                                    break
                            if freq_range_str != "N/A":
                                break
                        if freq_range_str != "N/A":
                            break
                    summary_rows.append({
                        'frequency_band': band_name, 'freq_range_Hz': freq_range_str,
                        'measure': measure.replace('_statistics', '').upper(),
                        'mean': stats['mean'], 'std': stats['std'], 'median': stats['median'],
                        'min': stats['min'], 'max': stats['max'], 'n_samples': stats['n_samples']
                    })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            csv_path = os.path.join(output_folder, f'correlation_summary_{dataset_name}_{timestamp}.csv')
            summary_df.to_csv(csv_path, index=False)
            print(f"Results saved: {os.path.basename(pickle_path)}, {os.path.basename(csv_path)}")
        else:
            print(f"Results saved: {os.path.basename(pickle_path)}")


# Main Processing Functions

def analyze_single_dataset_correlations(segments_file_path: str, output_folder: str = None) -> Dict:
    """Analyze spike-EEG correlations for a single dataset"""
    print(f"\nLoading dataset: {segments_file_path}")
    
    with open(segments_file_path, 'rb') as f:
        segments_data = pickle.load(f)
    
    dataset_name = segments_data.get('dataset_name', 'unknown')
    analyzer = SpikeTriggeredAnalyzer()
    results = analyzer.analyze_dataset_correlations(segments_data)
    
    if output_folder is None:
        base_dir = os.path.dirname(segments_file_path)
        output_folder = os.path.join(base_dir, 'correlation_analysis')
    os.makedirs(output_folder, exist_ok=True)
    
    analyzer.save_analysis_results(results, output_folder)
    analyzer.visualize_correlation_results(results, output_folder)
    
    print(f"Analysis complete: {dataset_name}\nResults: {output_folder}\n")
    return results


def analyze_all_datasets_correlations(base_path: str, dataset_prefix: str = 'path2_') -> Dict:
    """Analyze spike-EEG correlations for all datasets in base directory"""
    print(f"\nBatch processing: {base_path} (prefix: {dataset_prefix})")
    
    datasets = sorted([d for d in glob.glob(os.path.join(base_path, f'{dataset_prefix}*')) 
                      if os.path.isdir(d)])
    
    if not datasets:
        print(f"No datasets found with prefix '{dataset_prefix}'")
        return {}
    
    print(f"Found {len(datasets)} datasets\n")
    all_results = {}
    
    for idx, dataset_path in enumerate(datasets, 1):
        dataset_name = os.path.basename(dataset_path)
        print(f"\n[{idx}/{len(datasets)}] Processing: {dataset_name}")
        
        segments_pattern = os.path.join(dataset_path, 'neural_movement_segmentation', 'segments_*.pkl')
        segments_files = glob.glob(segments_pattern)
        
        if not segments_files:
            print(f"No segmented data found")
            all_results[dataset_name] = {'success': False, 'error': 'No segmented data file found'}
            continue
        
        segments_file = sorted(segments_files)[-1]
        output_folder = os.path.join(dataset_path, 'correlation_analysis')
        
        try:
            results = analyze_single_dataset_correlations(segments_file, output_folder)
            all_results[dataset_name] = {'success': True, 'results': results, 'output_folder': output_folder}
            print(f"✓ Success: {dataset_name}")
        except Exception as e:
            print(f"Error: {e}")
            all_results[dataset_name] = {'success': False, 'error': str(e)}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_summary_path = os.path.join(base_path, f'correlation_analysis_master_summary_{timestamp}.pkl')
    with open(master_summary_path, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    successful = sum(1 for r in all_results.values() if r.get('success', False))
    print(f"\nBatch complete: {successful}/{len(datasets)} successful")
    print(f"Master summary: {master_summary_path}\n")
    
    return all_results

# Example Usage

if __name__ == "__main__":
    BASE_PATH = '/home/ntk/exo/ExoPulse/simulation/mocap_EMG_EEG_data/20241225'
    
    single_dataset = os.path.join(BASE_PATH, 'path2_02')
    if os.path.exists(single_dataset):
        segments_pattern = os.path.join(single_dataset, 'neural_movement_segmentation', 'segments_*.pkl')
        segments_files = glob.glob(segments_pattern)
        if segments_files:
            print("\nProcessing single dataset: path2_02")
            output_path = os.path.join(single_dataset, 'correlation_analysis')
            results = analyze_single_dataset_correlations(segments_files[0], output_path)
            print("Single dataset analysis completed!")
        else:
            print("No segmented data found for path2_02")
    
    # all_results = analyze_all_datasets_correlations(BASE_PATH, dataset_prefix='path2_')

# %%