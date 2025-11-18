# %%
"""
Continuous Coherence Analysis Module for EEG-EMG Correlation
Performs coherence analysis between EEG signals and EMG-derived CWF components
across different movement states and frequency bands.

Key Features:
- Magnitude-squared coherence (MSC) calculation converted to Coherence Magnitude
- Multi-frequency band analysis (mu/beta/low_gamma for motor; alpha for visual)
- Processes all 32 EEG channels for complete spatial topology mapping
- Target channels (motor + visual) used for statistical summary and key comparisons
- **Simplified time alignment using relative time coordinates (0-based) from segmented data**

Coherence measures the linear correlation between two signals as a function of frequency,
ranging from 0 (no correlation) to 1 (perfect correlation).

Time Coordinate System (UPDATED):
- All segmented data uses relative time coordinates starting from 0
- EEG and CWF signals within each movement segment share the same timebase
- Time alignment is automatic and requires minimal interpolation
- CWF is pre-resampled to EEG sampling rate (500Hz) in neural_data_movement_segmentor
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Any, Optional
import os
import pickle
from datetime import datetime
import warnings

class CoherenceAnalyzer:
    
    def __init__(self, eeg_fs: float = 500.0, emg_fs: float = 1000.0):
        self.eeg_fs = eeg_fs
        self.emg_fs = emg_fs
        
        self.motor_freq_bands = {
            'delta': (1, 4), 'theta': (4, 8), 'mu': (8, 12),
            'beta': (12, 30), 'low_gamma': (30, 46)
        }
        
        self.visual_freq_bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12),
            'beta': (12, 30), 'low_gamma': (30, 46)
        }
        
        self.motor_channels = ['Cz', 'C3', 'C4', 'FCz', 'FC3', 'FC4', 'CPz', 'CP3', 'CP4']
        self.visual_channels = ['O1', 'O2', 'Oz']
        self.target_channels = self.motor_channels + self.visual_channels
        self.movement_states = ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']
        
        self.nperseg = 256
        self.noverlap = 128
        self.nfft = 512
        self.freq_max = 50.0
        
        self.motor_band_order = ['delta', 'theta', 'mu', 'beta', 'low_gamma']
        self.visual_band_order = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']
        
        self.band_colors = {
            'delta': '#1f77b4', 'theta': '#ff7f0e', 'mu': '#2ca02c',
            'alpha': '#2ca02c', 'beta': '#d62728', 'low_gamma': '#9467bd'
        }
    
    def _get_freq_bands_for_channel(self, channel_name: str) -> Dict[str, Tuple[float, float]]:
        return self.visual_freq_bands if channel_name in self.visual_channels else self.motor_freq_bands
    
    def _get_band_order_for_channel(self, channel_name: str) -> List[str]:
        return self.visual_band_order if channel_name in self.visual_channels else self.motor_band_order
    
    def align_and_truncate(self, eeg_time: np.ndarray, cwf_time: np.ndarray,
                          eeg_signal: np.ndarray, cwf_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if eeg_signal.ndim > 1 or cwf_signal.ndim > 1:
            raise ValueError(f"Signals must be 1D, got EEG: {eeg_signal.shape}, CWF: {cwf_signal.shape}")
        
        if len(eeg_time) == len(cwf_time) and np.allclose(eeg_time, cwf_time, rtol=1e-6, atol=1e-9):
            if len(eeg_signal) == len(eeg_time) and len(cwf_signal) == len(cwf_time):
                print(f"      Aligned: {len(eeg_time)} samples, duration={eeg_time[-1]:.3f}s")
                return eeg_signal, cwf_signal, eeg_time
            min_len = min(len(eeg_time), len(eeg_signal), len(cwf_signal))
            print(f"      Truncating to {min_len} samples")
            return eeg_signal[:min_len], cwf_signal[:min_len], eeg_time[:min_len]
        
        print(f"      Warning: Time mismatch - EEG: {len(eeg_time)}, CWF: {len(cwf_time)}")
        time_start = max(eeg_time[0], cwf_time[0])
        time_end = min(eeg_time[-1], cwf_time[-1])
        
        if time_end <= time_start:
            raise ValueError(f"No overlap: EEG=[{eeg_time[0]:.3f}, {eeg_time[-1]:.3f}], CWF=[{cwf_time[0]:.3f}, {cwf_time[-1]:.3f}]")
        
        eeg_mask = (eeg_time >= time_start) & (eeg_time <= time_end)
        cwf_mask = (cwf_time >= time_start) & (cwf_time <= time_end)
        eeg_aligned = eeg_signal[eeg_mask]
        eeg_time_aligned = eeg_time[eeg_mask]
        
        if len(eeg_time_aligned) == 0:
            raise ValueError(f"No EEG samples in overlap [{time_start:.3f}, {time_end:.3f}]")
        
        cwf_time_overlap = cwf_time[cwf_mask]
        if len(eeg_time_aligned) == len(cwf_time_overlap) and np.allclose(eeg_time_aligned, cwf_time_overlap, rtol=1e-6):
            cwf_aligned = cwf_signal[cwf_mask]
        else:
            f_interp = interp1d(cwf_time, cwf_signal, kind='linear', bounds_error=False, fill_value='extrapolate')
            cwf_aligned = f_interp(eeg_time_aligned)
            if np.any(np.isnan(cwf_aligned)):
                cwf_aligned = np.nan_to_num(cwf_aligned, nan=0.0)
        
        min_len = min(len(eeg_aligned), len(cwf_aligned))
        return eeg_aligned[:min_len], cwf_aligned[:min_len], eeg_time_aligned[:min_len]  
    
    def compute_coherence(self, signal1: np.ndarray, signal2: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        # calculating MSC
        freqs, Cxy_squared = signal.coherence(signal1, signal2, fs=fs, 
                                            nperseg=self.nperseg,
                                            noverlap=self.noverlap,
                                            nfft=self.nfft)

        # Rationale for using Coherence Magnitude instead of MSC:
        # 1. Interpretability: Cxy is analogous to Pearson correlation coefficient (r),
        #    while MSC is analogous to coefficient of determination (r²)
        # 2. Value comparison: Cxy provides more intuitive comparison between conditions
        #    Example: MSC = 0.64 → Cxy = 0.8 (easier to interpret as "80% correlation")
        # 3. Literature consistency: Our reference literature ("Corticomuscular coherence 
        #    existed at the single motor unit level") reports coherence magnitude
        # 4. Mathematical equivalence: Both metrics are valid and monotonically related
        #    (Cxy = √MSC), preserving relative differences

        # Convert MSC (Cxy²) to Coherence Magnitude (Cxy) by taking square root 
        Cxy = np.sqrt(Cxy_squared)
    
        freq_mask = freqs <= self.freq_max
        return freqs[freq_mask], Cxy[freq_mask]
    
    def extract_band_metrics(self, freqs: np.ndarray, values: np.ndarray,
                            freq_bands: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
        band_stats = {}
        for band_name, (f_low, f_high) in freq_bands.items():
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            band_values = values[band_mask]
            if len(band_values) > 0:
                band_stats[band_name] = {
                    'mean': float(np.mean(band_values)),
                    'max': float(np.max(band_values)),
                    'std': float(np.std(band_values))
                }
            else:
                band_stats[band_name] = {'mean': 0.0, 'max': 0.0, 'std': 0.0}
        return band_stats
    
    def analyze_pair(self, eeg_signal: np.ndarray, cwf_signal: np.ndarray,
                    eeg_time: np.ndarray, cwf_time: np.ndarray,
                    eeg_channel: str, cwf_label: str, comp_idx: int,
                    movement_state: str) -> Dict[str, Any]:
        try:
            eeg_aligned, cwf_aligned, common_time = self.align_and_truncate(
                eeg_time, cwf_time, eeg_signal, cwf_signal
            )
            
            min_samples = self.nperseg * 2
            if len(eeg_aligned) < min_samples:
                print(f"      ⚠ Segment too short ({len(eeg_aligned)} < {min_samples}), skipping...")
                return None
            
            freqs, coherence = self.compute_coherence(eeg_aligned, cwf_aligned, self.eeg_fs)
            freq_bands = self._get_freq_bands_for_channel(eeg_channel)
            band_stats = self.extract_band_metrics(freqs, coherence, freq_bands)
            
            return {
                'eeg_channel': eeg_channel,
                'cwf_label': cwf_label,
                'component': comp_idx,
                'movement_state': movement_state,
                'freqs': freqs,
                'coherence': coherence,
                'band_stats': band_stats,
                'n_samples': len(eeg_aligned),
                'duration': float(common_time[-1] - common_time[0])
            }
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None
    
    def analyze_dataset(self, segments_data: Dict) -> Dict[str, Any]:
        """
        Analyze coherence for entire dataset
        
        Args:
            segments_data: Dictionary containing segmented data from neural_data_movement_segmentor
        
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*80)
        print("COHERENCE ANALYSIS - Dataset Processing")
        print("="*80)
        
        results = {
            'dataset_info': {
                'timestamp': datetime.now().isoformat(),
                'eeg_fs': self.eeg_fs,
                'emg_fs': self.emg_fs,
                'parameters': {
                    'nperseg': self.nperseg,
                    'noverlap': self.noverlap,
                    'nfft': self.nfft,
                    'freq_max': self.freq_max
                }
            },
            'movement_states': {}
        }
        
        # Get EEG and EMG data
        eeg_segments = segments_data.get('eeg_segments', {})
        emg_segments = segments_data.get('emg_segments', {})
        
        # Process each movement state
        for state_name in self.movement_states:
            print(f"\n{'='*80}")
            print(f"Processing Movement State: {state_name.upper().replace('_', ' ')}")
            print(f"{'='*80}")
            
            state_results = {}
            
            # Process each EEG file
            for file_name, file_data in eeg_segments.items():
                if state_name not in file_data:
                    continue
                
                state_data = file_data[state_name]
                if state_data is None:
                    continue
                    
                eeg_data = state_data['data']
                eeg_time = state_data['time']
                eeg_channels = state_data.get('channels', [])
                
                # Handle data shape
                if eeg_data.ndim == 1:
                    eeg_data = eeg_data.reshape(1, -1)
                elif eeg_data.ndim == 2:
                    # EEG data should be (n_samples, n_channels) or (n_channels, n_samples)
                    # Check which dimension matches the time vector length
                    if eeg_data.shape[0] == len(eeg_time):
                        # Data is (n_samples, n_channels), transpose to (n_channels, n_samples)
                        eeg_data = eeg_data.T
                
                # Process each EEG channel (process ALL channels, not just target_channels)
                for ch_idx, channel_name in enumerate(eeg_channels):
                    # Skip non-standard channel names (like 'Time' or empty strings)
                    if not channel_name or channel_name.lower() == 'time':
                        continue
                    
                    print(f"\n  EEG Channel: {channel_name}")
                    
                    if channel_name not in state_results:
                        state_results[channel_name] = {}
                    
                    # Extract channel data (should be 1D time series)
                    if eeg_data.ndim == 2:
                        eeg_channel_data = eeg_data[ch_idx, :]
                    else:
                        eeg_channel_data = eeg_data.ravel()
                    
                    # Validate channel data length
                    if len(eeg_channel_data) != len(eeg_time):
                        print(f"    ⚠ Warning: EEG channel data length ({len(eeg_channel_data)}) != time length ({len(eeg_time)})")
                        min_len = min(len(eeg_channel_data), len(eeg_time))
                        eeg_channel_data = eeg_channel_data[:min_len]
                        eeg_time_used = eeg_time[:min_len]
                    else:
                        eeg_time_used = eeg_time
                    
                    # Process each mode (single/multi)
                    for mode in ['single', 'multi']:
                        if mode not in emg_segments:
                            continue
                        
                        print(f"    Mode: {mode}")
                        
                        # Process each EMG channel/label
                        for emg_label, emg_data_dict in emg_segments[mode].items():
                            cwf_components = emg_data_dict.get('CWF_comp', {})
                            
                            if state_name not in cwf_components:
                                continue
                            
                            cwf_state_data = cwf_components[state_name]
                            if cwf_state_data is None:
                                continue
                                
                            cwf_data = cwf_state_data['data']
                            cwf_time = cwf_state_data['time']
                            
                            # Handle data shape
                            if cwf_data.ndim == 1:
                                cwf_data = cwf_data.reshape(1, -1)
                            elif cwf_data.ndim == 2:
                                # CWF data should be (n_components, n_samples)
                                # Check which dimension matches the time vector length
                                if cwf_data.shape[0] == len(cwf_time):
                                    # Data is (n_samples, n_components), transpose to (n_components, n_samples)
                                    cwf_data = cwf_data.T
                            
                            n_components = cwf_data.shape[0]
                            
                            # Process each CWF component
                            for comp_idx in range(n_components):
                                analysis_key = f"{mode}_{emg_label}_comp{comp_idx+1}"
                                
                                # Extract component data (should be 1D time series)
                                if cwf_data.ndim == 2:
                                    cwf_component = cwf_data[comp_idx, :]
                                else:
                                    cwf_component = cwf_data.ravel()
                                
                                # Validate component data length
                                if len(cwf_component) != len(cwf_time):
                                    print(f"      ⚠ Warning: CWF component length ({len(cwf_component)}) != time length ({len(cwf_time)})")
                                    min_len = min(len(cwf_component), len(cwf_time))
                                    cwf_component = cwf_component[:min_len]
                                    cwf_time_used = cwf_time[:min_len]
                                else:
                                    cwf_time_used = cwf_time
                                
                                print(f"      Analyzing: {analysis_key}")
                                
                                # Perform coherence analysis
                                result = self.analyze_pair(
                                    eeg_channel_data, cwf_component,
                                    eeg_time_used, cwf_time_used,
                                    channel_name, emg_label, comp_idx + 1,
                                    state_name
                                )
                                
                                if result is not None:
                                    state_results[channel_name][analysis_key] = result
                                    print(f"        ✓ Completed: mean_coh={np.mean(result['coherence']):.3f}")
                                else:
                                    print(f"        ✗ Failed")
            
            results['movement_states'][state_name] = state_results
        
        # Compute summary statistics
        print(f"\n{'='*80}")
        print("Computing Summary Statistics...")
        print(f"{'='*80}")
        results['summary'] = self.compute_summary_statistics(results)
        
        return results
    
    def compute_summary_statistics(self, results: Dict) -> Dict[str, Any]:
        summary = {'band_statistics': {}}
        
        for band_name in ['delta', 'theta', 'mu', 'alpha', 'beta', 'low_gamma']:
            summary['band_statistics'][band_name] = {'coherence_values': []}
        
        for state_name, state_data in results.get('movement_states', {}).items():
            for channel_name, channel_data in state_data.items():
                for analysis_key, analysis_data in channel_data.items():
                    band_stats = analysis_data.get('band_stats', {})
                    for band_name, stats in band_stats.items():
                        if band_name in summary['band_statistics']:
                            summary['band_statistics'][band_name]['coherence_values'].append(stats.get('mean', 0.0))
        
        for band_name in summary['band_statistics'].keys():
            values = summary['band_statistics'][band_name]['coherence_values']
            if len(values) > 0:
                summary['band_statistics'][band_name].update({
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'n': len(values)
                })
            del summary['band_statistics'][band_name]['coherence_values']
        
        return summary
    
    def save_results(self, results: Dict, output_folder: str):
        """
        Save analysis results to disk
        
        Args:
            results: Results dictionary
            output_folder: Output directory path
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Save full results as pickle
        pickle_path = os.path.join(output_folder, 'coherence_results.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Saved results pickle: {pickle_path}")
        
        # Save summary statistics as CSV
        summary = results.get('summary', {})
        band_stats = summary.get('band_statistics', {})
        
        if band_stats:
            rows = []
            for band_name, band_data in band_stats.items():
                row = {'band': band_name}
                for key, value in band_data.items():
                    if not isinstance(value, list):
                        row[key] = value
                rows.append(row)
            
            df = pd.DataFrame(rows)
            csv_path = os.path.join(output_folder, 'coherence_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved summary CSV: {csv_path}")
    
    def visualize_results(self, results: Dict, output_folder: str):
        """
        Generate all visualizations
        
        Args:
            results: Results dictionary
            output_folder: Output directory path
        """
        print(f"\n{'='*80}")
        print("Generating Visualizations...")
        print(f"{'='*80}")
        
        movement_states = results.get('movement_states', {})
        
        # Generate spectrum plots (like Fig 9C: all components for same EMG channel)
        self._plot_coherence_spectra_by_component(movement_states, output_folder)
        
        # Generate topology maps (like Fig 9D: spatial distribution by frequency bands)
        self._plot_coherence_topology_maps(movement_states, output_folder)
        
        print(f"\n✓ All visualizations saved to: {output_folder}")
    
    def _plot_coherence_spectra_by_component(self, movement_states: Dict, output_folder: str):
        """
        Plot coherence spectra following Fig 9C logic:
        - One figure per EMG channel
        - Each subplot shows one EEG channel
        - All CWF components plotted as different colored lines with legend
        - No frequency band markers
        - Title positioned above subplots to avoid overlap
        """
        print("\n  → Plotting Coherence Spectra by Component...")
        
        for state_name, state_data in movement_states.items():
            state_title = state_name.replace('_', ' ').title()
            
            # Reorganize data: by mode_emg_channel -> eeg_channel -> components
            # Keep track of mode for folder structure
            data_by_mode_emg = {}
            
            for eeg_channel, channel_data in state_data.items():
                for analysis_key, analysis_data in channel_data.items():
                    if '_' not in analysis_key:
                        continue
                    
                    # Parse key: mode_emgchannel_compN
                    parts = analysis_key.split('_')
                    if len(parts) < 3:
                        continue
                    
                    # Extract mode, EMG channel and component
                    mode = parts[0]  # single, multi, etc.
                    comp = parts[-1]  # compN
                    emg_channel = '_'.join(parts[1:-1])
                    mode_emg_key = f"{mode}_{emg_channel}"
                    
                    if mode_emg_key not in data_by_mode_emg:
                        data_by_mode_emg[mode_emg_key] = {
                            'mode': mode,
                            'emg_channel': emg_channel,
                            'eeg_data': {}
                        }
                    
                    if eeg_channel not in data_by_mode_emg[mode_emg_key]['eeg_data']:
                        data_by_mode_emg[mode_emg_key]['eeg_data'][eeg_channel] = {}
                    
                    data_by_mode_emg[mode_emg_key]['eeg_data'][eeg_channel][comp] = analysis_data
            
            # Define colors for different components (matching reference figure style)
            comp_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Create one figure per mode-EMG combination showing all components
            for mode_emg_key in sorted(data_by_mode_emg.keys()):
                mode_emg_data = data_by_mode_emg[mode_emg_key]
                mode = mode_emg_data['mode']
                emg_channel = mode_emg_data['emg_channel']
                eeg_channels_dict = mode_emg_data['eeg_data']
                
                # Separate motor and visual channels
                motor_channels_present = [ch for ch in self.motor_channels if ch in eeg_channels_dict]
                visual_channels_present = [ch for ch in self.visual_channels if ch in eeg_channels_dict]
                all_channels = motor_channels_present + visual_channels_present
                
                if not all_channels:
                    continue
                
                # Create subplot grid (3 columns, like reference Fig 9C)
                n_channels = len(all_channels)
                n_cols = 3
                n_rows = int(np.ceil(n_channels / n_cols))
                
                # Create figure without constrained_layout to better control title position
                fig = plt.figure(figsize=(15, 3.5 * n_rows))
                
                # Add main title at the top with manual spacing
                title_text = f'Coherence Spectrum: {emg_channel} (All Components) | {state_title} | {mode.title()} Mode'
                fig.suptitle(title_text, fontsize=13, fontweight='bold')
                plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
                
                # Create subplots
                for idx, eeg_channel in enumerate(all_channels):
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                    
                    # Get all components for this EEG channel
                    components_data = eeg_channels_dict[eeg_channel]
                    
                    # Sort components by number (comp1, comp2, comp3, ...)
                    comp_names = sorted(components_data.keys(), 
                                       key=lambda x: int(x.replace('comp', '')) if 'comp' in x else 0)
                    
                    # Plot each component as a line (like 10%MVC, 20%MVC in reference)
                    for comp_idx, comp_name in enumerate(comp_names):
                        analysis_data = components_data[comp_name]
                        freqs = analysis_data['freqs']
                        coherence = analysis_data['coherence']
                        
                        # Select color
                        color = comp_colors[comp_idx % len(comp_colors)]
                        
                        # Simple line plot without fill_between or markers (matching reference)
                        ax.plot(freqs, coherence, color=color, linewidth=1.5, 
                               alpha=0.85, label=comp_name)
                    
                    # Formatting to match reference figure style
                    ax.set_xlabel('Frequency (Hz)', fontsize=9)
                    ax.set_ylabel('Coherence Spectrum', fontsize=9)
                    ax.set_title(eeg_channel, fontsize=10, fontweight='bold', pad=6)
                    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
                    ax.set_xlim([0, self.freq_max])
                    ax.set_ylim([0, 1])
                    ax.tick_params(labelsize=8)
                    
                    # Add legend (like reference figure with 10%MVC, 20%MVC)
                    if len(comp_names) > 0:
                        ax.legend(loc='upper right', fontsize=8, framealpha=0.95, 
                                 edgecolor='gray', fancybox=False)
                
                # Save figure with state/mode/Spectrum structure
                spectra_folder = os.path.join(output_folder, state_name, mode, 'Spectrum')
                os.makedirs(spectra_folder, exist_ok=True)
                output_file = os.path.join(spectra_folder, f'spectrum_{emg_channel}.png')
                try:
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"      ✓ Saved: spectrum_{emg_channel}.png")
                except Exception as e:
                    print(f"      ⚠ Warning saving {output_file}: {e}")
                    try:
                        plt.savefig(output_file, dpi=150)
                        print(f"      ✓ Saved with lower DPI: spectrum_{emg_channel}.png")
                    except:
                        pass
                
                plt.close(fig)
        
        print("    ✓ Coherence spectra by component completed")
    
    def _plot_coherence_topology_maps(self, movement_states: Dict, output_folder: str):
        print("\n  Generating coherence topology maps...")
        saved_count = 0
        from scipy.interpolate import Rbf
        
        channel_positions = {
            'Fp1': (-0.20, 0.65), 'Fp2': (0.20, 0.65),
            'AF3': (-0.25, 0.55), 'AF4': (0.25, 0.55),
            'F7': (-0.60, 0.40), 'F3': (-0.35, 0.40), 'Fz': (0.0, 0.40), 'F4': (0.35, 0.40), 'F8': (0.60, 0.40),
            'FT7': (-0.63, 0.20), 'FT8': (0.63, 0.20),
            'FC3': (-0.35, 0.20), 'FCz': (0.0, 0.20), 'FC4': (0.35, 0.20),
            'T7': (-0.65, 0.0), 'C3': (-0.35, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.35, 0.0), 'T8': (0.65, 0.0),
            'TP7': (-0.63, -0.20), 'TP8': (0.63, -0.20),
            'CP3': (-0.35, -0.20), 'CPz': (0.0, -0.20), 'CP4': (0.35, -0.20),
            'P7': (-0.60, -0.40), 'P3': (-0.35, -0.40), 'Pz': (0.0, -0.40), 'P4': (0.35, -0.40), 'P8': (0.60, -0.40),
            'O1': (-0.20, -0.60), 'Oz': (0.0, -0.60), 'O2': (0.20, -0.60),
        }
        
        for state_name, state_data in movement_states.items():
            state_title = state_name.replace('_', ' ').title()
            data_by_mode_emg = {}
            
            for eeg_channel, channel_data in state_data.items():
                for analysis_key, analysis_data in channel_data.items():
                    if '_' not in analysis_key:
                        continue
                    parts = analysis_key.split('_')
                    if len(parts) < 3:
                        continue
                    
                    mode = parts[0]
                    emg_channel = '_'.join(parts[1:-1])
                    comp = parts[-1]
                    mode_emg_key = f"{mode}_{emg_channel}"
                    
                    if mode_emg_key not in data_by_mode_emg:
                        data_by_mode_emg[mode_emg_key] = {'mode': mode, 'emg_channel': emg_channel, 'components': {}}
                    if comp not in data_by_mode_emg[mode_emg_key]['components']:
                        data_by_mode_emg[mode_emg_key]['components'][comp] = {}
                    data_by_mode_emg[mode_emg_key]['components'][comp][eeg_channel] = analysis_data
            
            for mode_emg_key, emg_data in data_by_mode_emg.items():
                components = emg_data['components']
                if not components:
                    continue
                
                mode = emg_data['mode']
                emg_channel = emg_data['emg_channel']
                comp_names = sorted(components.keys(), key=lambda x: int(x.replace('comp', '')))
                n_components = len(comp_names)
                band_order = self.motor_band_order
                n_bands = len(band_order)
                
                # Calculate global vmin/vmax for unified colorbar across all subplots
                all_coherence_values = []
                for comp_name in comp_names:
                    for band_name in band_order:
                        comp_data = components[comp_name]
                        for eeg_ch, analysis_data in comp_data.items():
                            if eeg_ch not in channel_positions:
                                continue
                            band_stats = analysis_data.get('band_stats', {})
                            actual_band = band_name
                            if band_name == 'mu' and eeg_ch in self.visual_channels:
                                actual_band = 'alpha'
                            if actual_band in band_stats:
                                all_coherence_values.append(band_stats[actual_band]['mean'])
                
                vmin = 0
                vmax = np.percentile(all_coherence_values, 95) if all_coherence_values else 0.3
                
                # Create figure: rows = components, cols = bands
                fig, axes = plt.subplots(n_components, n_bands, 
                                       figsize=(4 * n_bands, 4 * n_components))
                
                # Create one colorbar for entire figure (shared scale)
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                
                if n_components == 1 and n_bands == 1:
                    axes = np.array([[axes]])
                elif n_components == 1:
                    axes = axes.reshape(1, -1)
                elif n_bands == 1:
                    axes = axes.reshape(-1, 1)
                
                fig.suptitle(f'Coherence Topology Maps: {emg_channel} | '
                           f'{state_title} | {mode.title()} Mode',
                           fontsize=16, fontweight='bold')
                plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
                
                for comp_idx, comp_name in enumerate(comp_names):
                    for band_idx, band_name in enumerate(band_order):
                        ax = axes[comp_idx, band_idx]
                        comp_data = components[comp_name]
                        
                        coherence_values = {}
                        for eeg_ch, analysis_data in comp_data.items():
                            if eeg_ch not in channel_positions:
                                continue
                            band_stats = analysis_data.get('band_stats', {})
                            actual_band = band_name
                            if band_name == 'mu' and eeg_ch in self.visual_channels:
                                actual_band = 'alpha'
                            if actual_band in band_stats:
                                coherence_values[eeg_ch] = band_stats[actual_band]['mean']
                        
                        if not coherence_values:
                            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                            ax.axis('off')
                            continue
                        
                        # Head outline
                        circle = plt.Circle((0, 0), 0.7, fill=False, color='black', linewidth=2)
                        ax.add_patch(circle)
                        
                        # Nose (triangle pointing UP - apex at top)
                        nose_x = [0, -0.1, 0.1, 0]
                        nose_y = [0.85, 0.7, 0.7, 0.85]  # apex (0, 0.85) pointing upward
                        ax.plot(nose_x, nose_y, 'k-', linewidth=2)
                        
                        # Ears
                        ear_left = plt.Circle((-0.7, 0), 0.08, fill=False, color='black', linewidth=2, clip_on=False)
                        ear_right = plt.Circle((0.7, 0), 0.08, fill=False, color='black', linewidth=2, clip_on=False)
                        ax.add_patch(ear_left)
                        ax.add_patch(ear_right)
                        
                        x_coords, y_coords, values, labels = [], [], [], []
                        for ch, pos in channel_positions.items():
                            if ch in coherence_values:
                                x_coords.append(pos[0])
                                y_coords.append(pos[1])
                                values.append(coherence_values[ch])
                                labels.append(ch)
                        
                        grid_x, grid_y = np.mgrid[-0.7:0.7:300j, -0.7:0.7:300j]
                        rbf = Rbf(x_coords, y_coords, values, function='multiquadric', smooth=0.1)
                        grid_z = rbf(grid_x, grid_y)
                        mask = (grid_x**2 + grid_y**2) > 0.7**2
                        grid_z[mask] = np.nan
                        
                        im = ax.contourf(grid_x, grid_y, grid_z, levels=100, 
                                       cmap='RdBu_r', norm=norm, alpha=0.95, extend='max')
                        
                        for x, y, label in zip(x_coords, y_coords, labels):
                            if label in self.motor_channels or label in self.visual_channels:
                                ax.text(x, y, label, fontsize=7, ha='center', va='center',
                                      fontweight='bold', color='black', zorder=6)
                        
                        if comp_idx == 0:
                            band_display = 'Alpha' if band_name == 'mu' else 'Low Gamma' if band_name == 'low_gamma' else band_name.replace('_', ' ').title()
                            ax.set_title(band_display, fontsize=11, fontweight='bold')
                        
                        if band_idx == 0:
                            ax.text(-1.3, 0, comp_name, fontsize=12, fontweight='bold', ha='center', va='center')
                        
                        ax.set_xlim([-0.9, 0.9])
                        ax.set_ylim([-0.9, 0.9])
                        ax.set_aspect('equal')
                        ax.axis('off')
                
                # Apply tight_layout first (before adding colorbar to avoid warning)
                plt.tight_layout(rect=[0, 0, 0.93, 0.95])
                
                # Add single colorbar for entire figure (after tight_layout)
                cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('Coherence', fontsize=11, fontweight='bold')
                cbar.ax.tick_params(labelsize=9)
                
                # Save figure with state/mode/TopologyMaps structure
                save_folder = os.path.join(output_folder, state_name, mode, 'TopologyMaps')
                os.makedirs(save_folder, exist_ok=True)
                filename = f'topology_{emg_channel}.png'
                save_path = os.path.join(save_folder, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Print save confirmation
                print(f"      ✓ Saved: {filename}")
                saved_count += 1
        
        print(f"    ✓ Coherence topology maps completed ({saved_count} files saved)")

# ==================== Main Processing Functions ====================

def analyze_single_dataset_coherence(segments_file_path: str, 
                                    output_folder: str = None) -> Dict:
    """
    Analyze EEG-CWF coherence for a single dataset
    
    Args:
        segments_file_path: Path to segmented data pickle file 
                           (output from neural_data_movement_segmentor)
        output_folder: Output directory for results and plots
        
    Returns:
        Analysis results dictionary
    """
    print(f"\n{'='*80}")
    print(f"LOADING DATASET")
    print(f"{'='*80}")
    print(f"File: {segments_file_path}")
    
    # Load segmented data
    with open(segments_file_path, 'rb') as f:
        segments_data = pickle.load(f)
    
    dataset_name = segments_data.get('dataset_name', 'unknown')
    print(f"Dataset: {dataset_name}")
    
    # Initialize analyzer
    analyzer = CoherenceAnalyzer()
    
    # Run coherence analysis
    results = analyzer.analyze_dataset(segments_data)
    
    # Determine output folder
    if output_folder is None:
        base_dir = os.path.dirname(segments_file_path)
        output_folder = os.path.join(base_dir, 'coherence_analysis')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Save results and create visualizations
    analyzer.save_results(results, output_folder)
    analyzer.visualize_results(results, output_folder)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_folder}\n")
    
    return results


def analyze_all_datasets_coherence(base_path: str, 
                                  dataset_prefix: str = 'path2_') -> Dict:
    """
    Analyze EEG-CWF coherence for all datasets in base directory
    Processes multiple datasets and generates master summary
    
    Args:
        base_path: Base directory containing dataset folders
        dataset_prefix: Prefix for dataset folder names (e.g., 'path2_')
        
    Returns:
        Dictionary of results for all datasets
    """
    import glob
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING DATASETS")
    print(f"{'='*80}")
    print(f"Base path: {base_path}")
    print(f"Dataset prefix: {dataset_prefix}")
    
    # Find all dataset folders
    datasets = sorted([d for d in glob.glob(os.path.join(base_path, f'{dataset_prefix}*')) 
                      if os.path.isdir(d)])
    
    if not datasets:
        print(f"No datasets found with prefix '{dataset_prefix}' in {base_path}")
        return {}
    
    print(f"\nFound {len(datasets)} datasets")
    
    all_results = {}
    
    # Process each dataset
    for idx, dataset_path in enumerate(datasets, 1):
        dataset_name = os.path.basename(dataset_path)
        print(f"\n{'='*80}")
        print(f"Processing dataset {idx}/{len(datasets)}: {dataset_name}")
        print(f"{'='*80}")
        
        # Look for segmented data file
        segments_pattern = os.path.join(dataset_path, 'neural_movement_segmentation', 
                                       'segments_*.pkl')
        segments_files = glob.glob(segments_pattern)
        
        if not segments_files:
            print(f"No segmented data found for {dataset_name}")
            all_results[dataset_name] = {
                'success': False,
                'error': 'No segmented data file found'
            }
            continue
        
        # Use most recent segments file
        segments_file = sorted(segments_files)[-1]
        output_folder = os.path.join(dataset_path, 'coherence_analysis')
        
        try:
            results = analyze_single_dataset_coherence(segments_file, output_folder)
            all_results[dataset_name] = {
                'success': True,
                'results': results,
                'output_folder': output_folder
            }
            print(f"✓ Successfully analyzed {dataset_name}")
            
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Save master summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_summary_path = os.path.join(base_path, 
                                      f'coherence_analysis_master_summary_{timestamp}.pkl')
    
    with open(master_summary_path, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Print summary
    successful = sum(1 for r in all_results.values() if r.get('success', False))
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {successful}/{len(datasets)} datasets")
    print(f"Master summary saved to: {master_summary_path}\n")
    
    return all_results

# ==================== Example Usage ====================

if __name__ == "__main__":
    import glob
    
    # Set base path to your data directory
    BASE_PATH = '/home/ntk/exo/ExoPulse/simulation/mocap_EMG_EEG_data/20241225'
    
    # Example 1: Process single dataset
    single_dataset = os.path.join(BASE_PATH, 'path2_02')
    
    if os.path.exists(single_dataset):
        segments_pattern = os.path.join(single_dataset, 'neural_movement_segmentation', 
                                       'segments_*.pkl')
        segments_files = glob.glob(segments_pattern)
        
        if segments_files:
            print(f"\n{'='*80}")
            print("PROCESSING SINGLE DATASET: path2_02")
            print(f"{'='*80}")
            
            output_path = os.path.join(single_dataset, 'coherence_analysis')
            results = analyze_single_dataset_coherence(segments_files[0], output_path)

            print("Single dataset coherence analysis completed!")
        else:
            print("No segmented data found for path2_02")

    # Example 2: Process all datasets (uncomment to use)
    # print(f"\n{'='*80}")
    # print("PROCESSING ALL DATASETS")
    # print(f"{'='*80}")
    # all_results = analyze_all_datasets_coherence(BASE_PATH, dataset_prefix='path2_')
    # print("\n✓ All datasets coherence analysis completed!")

# %%
