# %%
"""
Spike-Triggered Analysis Module for Multi-Modal Neural Data
Performs spike-triggered correlation analysis between EMG-derived spikes and EEG signals
across different movement states and frequency bands.

Key Features:
- Spike-Triggered Average (STA) analysis
- Pairwise Phase Consistency (PPC) calculation
- Multi-frequency band analysis (delta, theta, mu/alpha, beta, low gamma)
- Movement state-specific analysis
- Integration with neural_data_movement_segmentor data structure
- **Works with relative time coordinates (0-based) from segmented data**

EEG Frequency Bands:
- Delta: 1-4 Hz (deep sleep, unconscious processes)
- Theta: 4-8 Hz (drowsiness, meditation)
- Mu: 8-12 Hz (sensorimotor rhythm, for motor channels)
- Alpha: 8-12 Hz (visual attention, for visual channels)
- Beta: 12-30 Hz (motor control, active thinking)
- Low Gamma: 30-46 Hz (motor cortex activity, cognitive processing)

Target EEG Channels:
- Motor control: Cz, C3, C4, FCz, FC3, FC4, CPz, CP3, CP4 (use mu band)
- Visual processing (event trigger): O1, O2, Oz (use alpha band)

Time Coordinate System:
- All data uses relative time coordinates starting from 0 within each movement segment
- EEG and spike times are automatically aligned through shared relative timebase
- No manual time alignment or interpolation needed between modalities
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
    """
    Comprehensive spike-triggered analysis for EEG-EMG correlation studies
    Focuses on STA and PPC measures only
    """
    
    def __init__(self, eeg_fs: float = 500.0, emg_fs: float = 1000.0):
        """
        Initialize analyzer with sampling rates and analysis parameters
        
        Args:
            eeg_fs: EEG sampling frequency (Hz)
            emg_fs: EMG sampling frequency (Hz)
        """
        self.eeg_fs = eeg_fs
        self.emg_fs = emg_fs
        
        # Define frequency bands for MOTOR channels
        self.motor_freq_bands = {
            'delta': (1, 4),        # Deep sleep, unconscious
            'theta': (4, 8),        # Drowsiness, meditation
            'mu': (8, 12),          # Sensorimotor rhythm
            'beta': (12, 30),       # Motor control
            'low_gamma': (30, 46)   # Motor cortex activity
        }
        
        # Define frequency bands for VISUAL channels
        self.visual_freq_bands = {
            'delta': (1, 4),        # Deep sleep, unconscious
            'theta': (4, 8),        # Drowsiness, meditation
            'alpha': (8, 12),       # Visual attention, relaxed wakefulness
            'beta': (12, 30),       # Active thinking
            'low_gamma': (30, 46)   # Cognitive processing
        }
        
        # Motor control related EEG channels
        self.motor_channels = [
            'Cz', 'C3', 'C4',       # Central motor cortex
            'FCz', 'FC3', 'FC4',    # Frontal-central
            'CPz', 'CP3', 'CP4'     # Central-parietal
        ]
        
        # Visual processing channels (for event trigger analysis)
        self.visual_channels = ['O1', 'O2', 'Oz']
        
        # Combined target channels
        self.target_channels = self.motor_channels + self.visual_channels
        
        # Channel groups for organized visualization
        self.channel_groups = {
            'motor': self.motor_channels,
            'visual': self.visual_channels
        }
        
        # Movement states
        self.movement_states = ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']
        
        # Analysis window parameters (in seconds)
        self.sta_window = 0.5    # ±0.5s around spike for STA
        
        # Define frequency band order (low to high) - separate for motor/visual
        self.motor_band_order = ['delta', 'theta', 'mu', 'beta', 'low_gamma']
        self.visual_band_order = ['delta', 'theta', 'alpha', 'beta', 'low_gamma']
    
    def _get_freq_bands_for_channel(self, channel_name: str) -> Dict[str, Tuple[float, float]]:
        """
        Get appropriate frequency bands based on channel type
        
        Args:
            channel_name: EEG channel name
            
        Returns:
            Dictionary of frequency bands for this channel
        """
        if channel_name in self.visual_channels:
            return self.visual_freq_bands
        else:
            return self.motor_freq_bands
    
    def _get_band_display_name(self, channel_name: str, band_name: str) -> str:
        """
        Get display name for frequency band based on channel type
        
        Args:
            channel_name: EEG channel name
            band_name: Frequency band name
            
        Returns:
            Display name for the band
        """
        freq_bands = self._get_freq_bands_for_channel(channel_name)
        freq_range = freq_bands.get(band_name, (0, 0))
        return f"{band_name.title()} ({freq_range[0]}-{freq_range[1]} Hz)"
    
    def _get_band_order_for_channel(self, channel_name: str) -> List[str]:
        """
        Get frequency band order based on channel type
        
        Args:
            channel_name: EEG channel name
            
        Returns:
            List of band names in order (low to high frequency)
        """
        if channel_name in self.visual_channels:
            return self.visual_band_order
        else:
            return self.motor_band_order
        
    def reorganize_eeg_data(self, segments_data: Dict) -> Dict:
        """
        Reorganize EEG data from [eeg_file][state] to [state][channel] structure
        All time coordinates are RELATIVE (starting from 0) as provided by neural_data_movement_segmentor
        
        Args:
            segments_data: Original segmented data from neural_data_movement_segmentor
            
        Returns:
            Reorganized EEG data structure with format:
            {state: {channel: {data, time (relative), fs, time_format, absolute_start, absolute_end, ...}}}
        """
        reorganized_eeg = {}
        
        # Initialize structure for each movement state
        for state in self.movement_states:
            reorganized_eeg[state] = {}
        
        # Process each EEG file
        eeg_segments = segments_data.get('eeg_segments', {})
        
        for eeg_filename, file_segments in eeg_segments.items():
            print(f"Processing EEG file: {eeg_filename}")
            
            # Process each movement state for this file
            for state in self.movement_states:
                if state in file_segments and file_segments[state] is not None:
                    segment = file_segments[state]
                    
                    # Get channel information
                    channels = segment.get('channels', [])
                    data = segment['data']  # Shape: (n_samples, n_channels)
                    time = segment['time']
                    
                    print(f"  State {state}: {len(channels)} channels, {data.shape} data shape")
                    
                    # Extract data for each channel
                    for ch_idx, channel_name in enumerate(channels):
                        # Process target channels (motor + visual)
                        if channel_name in self.target_channels:
                            if data.ndim > 1 and ch_idx < data.shape[1]:
                                channel_data = data[:, ch_idx]
                            elif data.ndim == 1 and ch_idx == 0:
                                channel_data = data
                            else:
                                continue
                                
                            reorganized_eeg[state][channel_name] = {
                                'data': channel_data,
                                'time': time,  # RELATIVE time from neural_data_movement_segmentor
                                'fs': segment.get('fs', self.eeg_fs),
                                'source_file': eeg_filename,
                                'duration': segment.get('duration', 0),
                                'n_samples': len(channel_data),
                                'time_format': segment.get('time_format', 'relative'),  # Should be 'relative'
                                'absolute_start': segment.get('absolute_start', None),  # Reference only
                                'absolute_end': segment.get('absolute_end', None),      # Reference only
                                'data_start_time': segment.get('data_start_time', time[0])  # Kept for compatibility
                            }
                            
                            print(f"    Added channel {channel_name}: {len(channel_data)} samples")
        
        return reorganized_eeg
    
    def reorganize_spike_data(self, segments_data: Dict) -> Dict:
        """
        Reorganize spike data to be more accessible for analysis
        
        Args:
            segments_data: Original segmented data from neural_data_movement_segmentor
            
        Returns:
            Reorganized spike data structure with format:
            {mode: {state: {comp_key: {spike_times, n_spikes, ...}}}}
        """
        reorganized_spikes = {}
        
        spike_segments = segments_data.get('spike_segments', {})
        
        for mode in ['single', 'multi']:
            if mode not in spike_segments:
                continue
                
            reorganized_spikes[mode] = {}
            mode_data = spike_segments[mode]
            
            # Initialize structure for each movement state
            for state in self.movement_states:
                reorganized_spikes[mode][state] = {}
            
            # Process each channel
            for ch_label, ch_segments in mode_data.items():
                print(f"Processing spike channel: {ch_label} ({mode} mode)")
                
                # Process each component
                for comp_name, comp_segments in ch_segments.items():
                    comp_key = f"{ch_label}_{comp_name}"
                    
                    # Process each movement state
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
                                'channel': ch_label,
                                'component': comp_name,
                                'time_format': segment.get('time_format', 'relative'),
                                'absolute_start': segment.get('absolute_start', None),
                                'absolute_end': segment.get('absolute_end', None)
                            }
                            
                            print(f"  State {state}, {comp_key}: {segment['n_spikes']} spikes")
        
        return reorganized_spikes
    
    def bandpass_filter_eeg(self, eeg_data: np.ndarray, freq_band: Tuple[float, float], 
                       fs: float, filter_order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to EEG data for specific frequency band
        Uses Butterworth filter with safe parameter handling
        
        Args:
            eeg_data: EEG signal array (1D or 2D)
            freq_band: Tuple of (low_freq, high_freq) in Hz
            fs: Sampling frequency
            filter_order: Butterworth filter order
            
        Returns:
            Filtered EEG data with same shape as input
        """
        low_freq, high_freq = freq_band
        nyquist = fs / 2.0
    
        # Normalize frequencies and ensure valid range
        if high_freq >= nyquist * 0.99:  
            high_freq = nyquist * 0.95
            warnings.warn(f"High frequency adjusted to {high_freq:.2f} Hz (95% of Nyquist)")
    
        if low_freq <= 0:
            low_freq = 0.5
            warnings.warn(f"Low frequency adjusted to {low_freq} Hz")
        
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
    
        # Clip to valid range for butter filter
        low_norm = np.clip(low_norm, 0.001, 0.999)
        high_norm = np.clip(high_norm, 0.001, 0.999)
    
        if low_norm >= high_norm:
            raise ValueError(f"Invalid frequency band: {low_freq}-{high_freq} Hz")
    
        # Design Butterworth bandpass filter
        b, a = butter(filter_order, [low_norm, high_norm], btype='bandpass')
    
        # Apply filter with safe padding to reduce edge effects
        if eeg_data.ndim == 1:
            # 1D EEG data
            pad_length = max(1, min(len(eeg_data) // 2 - 1, 100))
            filtered_data = filtfilt(b, a, eeg_data, padlen=pad_length)
        else:
            # 2D EEG data (multi-channel)
            filtered_data = np.array([
                filtfilt(b, a, eeg_data[:, ch], 
                        padlen=max(1, min(eeg_data.shape[0] // 2 - 1, 100))) 
                for ch in range(eeg_data.shape[1])
            ]).T
        
        return filtered_data

    def extract_spike_triggered_segments(self, eeg_signal: np.ndarray, spike_times: np.ndarray,
                                       eeg_time: np.ndarray, window_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract EEG segments around spike times for spike-triggered analysis
        Works with RELATIVE time coordinates (0-based) from neural_data_movement_segmentor
        
        Args:
            eeg_signal: EEG signal array (1D)
            spike_times: Array of spike times in RELATIVE coordinates (seconds, starting from 0)
            eeg_time: EEG time vector in RELATIVE coordinates (seconds, starting from 0)
            window_size: Window size around spike (±window_size seconds)
            
        Returns:
            Tuple of (segments, relative_time_vector)
            segments: Array of shape (n_valid_spikes, n_samples_per_window)
            relative_time_vector: Time vector relative to spike (seconds, ±window_size)
        """
        if len(spike_times) == 0:
            return np.array([]), np.array([])
            
        # Calculate window parameters
        window_samples = int(window_size * self.eeg_fs)
        total_samples = 2 * window_samples + 1
        relative_time = np.linspace(-window_size, window_size, total_samples)
        
        segments = []
        valid_spikes = []
        
        for spike_time in spike_times:
            # Find closest EEG sample to spike time
            spike_idx = np.argmin(np.abs(eeg_time - spike_time))
            
            # Check if window fits within EEG data bounds
            start_idx = spike_idx - window_samples
            end_idx = spike_idx + window_samples + 1
            
            if start_idx >= 0 and end_idx <= len(eeg_signal):
                segment = eeg_signal[start_idx:end_idx]
                if len(segment) == total_samples:  # Ensure consistent length
                    segments.append(segment)
                    valid_spikes.append(spike_time)
        
        if len(segments) > 0:
            segments = np.array(segments)
        else:
            segments = np.array([]).reshape(0, total_samples)
            
        return segments, relative_time

    def compute_spike_triggered_average(self, eeg_segments: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute Spike-Triggered Average (STA) from EEG segments
        
        Args:
            eeg_segments: Array of EEG segments around spikes (n_spikes, n_samples)
            
        Returns:
            Dictionary containing:
            - mean: Average EEG response across spikes
            - std: Standard deviation across spikes
            - sem: Standard error of the mean
            - n_spikes: Number of spikes used
        """
        if eeg_segments.size == 0:
            return {'mean': np.array([]), 'std': np.array([]), 'sem': np.array([]), 'n_spikes': 0}
            
        n_spikes = eeg_segments.shape[0]
        
        # Compute statistics across spikes
        sta_mean = np.mean(eeg_segments, axis=0)
        sta_std = np.std(eeg_segments, axis=0)
        sta_sem = sta_std / np.sqrt(n_spikes) if n_spikes > 1 else np.zeros_like(sta_std)
        
        return {
            'mean': sta_mean,
            'std': sta_std,
            'sem': sta_sem,
            'n_spikes': n_spikes
        }
    
    def compute_instantaneous_phase(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase using Hilbert transform
        
        Args:
            signal: Input signal (1D array)
            
        Returns:
            Instantaneous phase array (radians, -π to π)
        """
        analytic_signal = hilbert(signal)
        phase = np.angle(analytic_signal)
        return phase

    def compute_pairwise_phase_consistency(self, phase_segments: np.ndarray) -> Dict[str, float]:
        """
        Compute pairwise phase consistency (PPC) for spike-triggered EEG phases
        PPC is an unbiased measure of phase synchronization
        Uses instantaneous phase at exact spike time (center of segment)
        
        Args:
            phase_segments: Phase segments around spikes (n_spikes, n_samples)

        Returns:
            Dictionary containing:
            - ppc: Pairwise phase consistency value (0-1)
            - n_pairs: Number of spike pairs used
            - n_spikes: Number of spikes
        """
        if phase_segments.size == 0:
            return {'ppc': 0.0, 'n_pairs': 0, 'n_spikes': 0}
        
        n_spikes, n_samples = phase_segments.shape
        center_idx = n_samples // 2
        
        # Extract instantaneous phase at exact spike time
        spike_phase_values = phase_segments[:, center_idx]
        
        N = len(spike_phase_values)
        if N < 2:
            return {'ppc': 0.0, 'n_pairs': 0, 'n_spikes': N}
        
        # Compute PPC using pairwise formula
        # PPC = (|Σ exp(i*φ)|² - N) / (N*(N-1))
        z = np.sum(np.exp(1j * spike_phase_values))
        numerator = np.abs(z)**2 - N
        denominator = N * (N - 1)
        ppc = numerator / denominator
        
        return {
            'ppc': float(np.clip(ppc, 0, 1)),  # Clip to valid range
            'n_pairs': N * (N - 1) // 2,
            'n_spikes': N
        }
    
    def analyze_channel_spike_correlation(self, eeg_data: np.ndarray, eeg_time: np.ndarray,
                                    spike_times: np.ndarray, channel_name: str,
                                    movement_state: str) -> Dict[str, Any]:
        """
        Perform comprehensive spike-triggered analysis for single EEG channel
        Computes STA and PPC for each frequency band
        Works with RELATIVE time coordinates (0-based) - no alignment needed
    
        Args:
            eeg_data: EEG signal (1D array)
            eeg_time: EEG time vector in RELATIVE coordinates (starting from 0)
            spike_times: Array of spike times in RELATIVE coordinates (starting from 0)
            channel_name: Name of EEG channel
            movement_state: Movement state identifier
        
        Returns:
            Dictionary containing STA and PPC results for each frequency band
        """
        
        # Get appropriate frequency bands for this channel
        freq_bands = self._get_freq_bands_for_channel(channel_name)
        channel_type = 'Visual' if channel_name in self.visual_channels else 'Motor'
    
        # ============== DEBUG: Initial Information ==============
        print(f"\n      {'='*60}")
        print(f"      DEBUG: Spike-EEG Correlation Analysis")
        print(f"      {'='*60}")
        print(f"      Channel: {channel_name} ({channel_type})")
        print(f"      Frequency Bands: {list(freq_bands.keys())}")
        print(f"      Movement State: {movement_state}")
        print(f"      EEG time range: [{eeg_time[0]:.3f}s, {eeg_time[-1]:.3f}s]")
        print(f"      EEG duration: {eeg_time[-1] - eeg_time[0]:.3f}s")
        print(f"      EEG samples: {len(eeg_data)}")
        print(f"      EEG sampling rate: {self.eeg_fs} Hz")
        print(f"      Number of spikes: {len(spike_times)}")
    
        if len(spike_times) > 0:
            print(f"      Spike time range: [{np.min(spike_times):.3f}s, {np.max(spike_times):.3f}s]")
            print(f"      First 5 spike times: {spike_times[:5]}")
            if len(spike_times) > 5:
                print(f"      Last 5 spike times: {spike_times[-5:]}")
    
        # Check spike alignment with EEG time range
        spikes_in_range = spike_times[(spike_times >= eeg_time[0]) & 
                                    (spike_times <= eeg_time[-1])]
        print(f"      Spikes within EEG range: {len(spikes_in_range)}/{len(spike_times)}")
    
        # Check spikes with sufficient window space
        sta_margin = self.sta_window
        valid_spikes_sta = spike_times[(spike_times >= eeg_time[0] + sta_margin) & 
                                    (spike_times <= eeg_time[-1] - sta_margin)]
        print(f"      Spikes with valid STA windows (±{sta_margin}s): {len(valid_spikes_sta)}/{len(spike_times)}")
    
        # PPC uses same margin as STA (to extract phase segments, but only center point is used)
        ppc_margin = sta_margin
        valid_spikes_ppc = spike_times[(spike_times >= eeg_time[0] + ppc_margin) & 
                                    (spike_times <= eeg_time[-1] - ppc_margin)]
        print(f"      Spikes with valid PPC segments (±{ppc_margin}s): {len(valid_spikes_ppc)}/{len(spike_times)}")
    
        # Identify problematic spikes
        if len(spikes_in_range) < len(spike_times):
            out_of_range = spike_times[(spike_times < eeg_time[0]) | (spike_times > eeg_time[-1])]
            print(f"      ⚠ WARNING: {len(out_of_range)} spikes outside EEG range:")
            print(f"         {out_of_range}")
    
        if len(valid_spikes_sta) == 0:
            print(f"      ⚠ WARNING: No spikes with sufficient STA window space!")
            print(f"         Need: spike_time ∈ [{eeg_time[0] + sta_margin:.3f}s, {eeg_time[-1] - sta_margin:.3f}s]")
    
        print(f"      {'='*60}\n")
    
        results = {
            'channel': channel_name,
            'channel_type': channel_type,
            'movement_state': movement_state,
            'n_spikes': len(spike_times),
            'bands': {}
        }
        
        # Analyze each frequency band (using channel-specific bands)
        for band_name, freq_range in freq_bands.items():
            try:
                print(f"      Analyzing {band_name} band ({freq_range[0]}-{freq_range[1]} Hz)...")
            
                # Filter EEG data for this frequency band
                filtered_eeg = self.bandpass_filter_eeg(eeg_data, freq_range, self.eeg_fs)
                print(f"        Filtered EEG shape: {filtered_eeg.shape}")
            
                # Extract spike-triggered segments for STA
                eeg_segments, relative_time = self.extract_spike_triggered_segments(
                    filtered_eeg, spike_times, eeg_time, self.sta_window
                )
            
                print(f"        STA segments extracted: {len(eeg_segments)}/{len(spike_times)}")
            
                if len(eeg_segments) == 0:
                    print(f"        ⚠ {band_name}: no valid segments extracted")
                    print(f"           Reason: All spikes too close to EEG boundaries (need ±{self.sta_window}s)")
                    continue
            
                print(f"        STA segment shape: {eeg_segments.shape}")
                print(f"        STA relative time range: [{relative_time[0]:.3f}s, {relative_time[-1]:.3f}s]")
            
                # Compute STA
                sta_results = self.compute_spike_triggered_average(eeg_segments)
                print(f"        STA computed: mean shape={sta_results['mean'].shape}, n_spikes={sta_results['n_spikes']}")
            
                # Compute phase for PPC analysis
                analytic_signal = hilbert(filtered_eeg)
                phase = np.angle(analytic_signal)
                print(f"        Phase signal shape: {phase.shape}")
            
                # Extract phase segments around spikes (same window as STA, but PPC uses center point only)
                phase_segments, _ = self.extract_spike_triggered_segments(
                    phase, spike_times, eeg_time, self.sta_window
                )
            
                print(f"        PPC segments extracted: {len(phase_segments)}/{len(spike_times)}")
            
                if len(phase_segments) > 0:
                    print(f"        PPC segment shape: {phase_segments.shape}")
            
                # Compute PPC
                ppc_results = self.compute_pairwise_phase_consistency(phase_segments)
                print(f"        PPC computed: value={ppc_results['ppc']:.4f}, n_pairs={ppc_results['n_pairs']}, n_spikes={ppc_results['n_spikes']}")
            
                # Store results for this frequency band
                results['bands'][band_name] = {
                    'sta': sta_results,
                    'ppc': ppc_results,
                    'n_valid_segments': len(eeg_segments),
                    'relative_time': relative_time,
                    'freq_range': freq_range
                }
            
                print(f"        ✓ Completed: {len(eeg_segments)} segments, PPC={ppc_results['ppc']:.3f}")
            
            except Exception as e:
                print(f"        ✗ Error in {band_name}: {e}")
                import traceback
                print(f"        Traceback:")
                traceback.print_exc()
                continue
    
        print(f"\n      Total bands analyzed: {len(results['bands'])}/{len(freq_bands)}\n")
    
        return results if results['bands'] else None
    
    def analyze_dataset_correlations(self, segments_data: Dict) -> Dict[str, Any]:
        """
        Analyze spike-EEG correlations for entire dataset across all movement states
        Uses relative time coordinates (0-based) from neural_data_movement_segmentor
        No manual time alignment needed - all modalities share the same relative timebase
        
        Args:
            segments_data: Segmented data dictionary from neural_data_movement_segmentor
            
        Returns:
            Comprehensive analysis results organized by state and channel
        """
        dataset_name = segments_data.get('dataset_name', 'unknown')
        print(f"\n{'='*70}")
        print(f"Analyzing spike-EEG correlations for dataset: {dataset_name}")
        print(f"{'='*70}")
        
        results = {
            'dataset_name': dataset_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'movement_states': {},
            'summary_statistics': {}
        }
        
        # Reorganize data structures for easier access
        print("\nReorganizing EEG data structure...")
        reorganized_eeg = self.reorganize_eeg_data(segments_data)
        
        print("Reorganizing spike data structure...")
        reorganized_spikes = self.reorganize_spike_data(segments_data)
        
        # Get available movement states
        available_states = segments_data.get('available_states', [])
        print(f"\nAvailable movement states: {available_states}")
        
        # Analyze each movement state
        for state in available_states:
            if state not in self.movement_states:
                continue
                
            print(f"\n{'-'*70}")
            print(f"Analyzing movement state: {state.replace('_', ' ').upper()}")
            print(f"{'-'*70}")
            results['movement_states'][state] = {}
            
            # Get EEG data for this state
            state_eeg = reorganized_eeg.get(state, {})
            if not state_eeg:
                print(f"  No EEG data found for state {state}")
                continue
                
            print(f"  Found EEG channels: {list(state_eeg.keys())}")
            
            # Analyze each EEG channel
            for channel in self.target_channels:  
                if channel not in state_eeg:
                    continue
                    
                print(f"\n  Processing EEG channel: {channel}")
                eeg_info = state_eeg[channel]
                eeg_data = eeg_info['data']
                eeg_time = eeg_info['time']
                
                # Get spike data for this state
                channel_spike_results = {}
                
                for mode in ['single', 'multi']:
                    if mode not in reorganized_spikes:
                        continue
                        
                    mode_spikes = reorganized_spikes[mode].get(state, {})
                    
                    for comp_name, comp_data in mode_spikes.items():
                        if 'spike_times' not in comp_data or len(comp_data['spike_times']) < 2:
                            continue
                            
                        analysis_key = f"{mode}_{comp_name}"
                        
                        # Get spike times (already in RELATIVE time coordinates from segmentor)
                        spike_times = np.asarray(comp_data['spike_times'], dtype=float)
                        
                        # Check time format to ensure consistency
                        time_format = comp_data.get('time_format', 'unknown')
                        if time_format != 'relative':
                            print(f"    ⚠ Warning: {analysis_key} has unexpected time_format='{time_format}', expected 'relative'")
                        
                        print(f"    {analysis_key}: {len(spike_times)} spikes (relative time)")
                        
                        # Verify spike times are in valid range (should start near 0 for relative time)
                        if len(spike_times) > 0:
                            spike_min, spike_max = np.min(spike_times), np.max(spike_times)
                            print(f"      Spike time range: [{spike_min:.3f}s, {spike_max:.3f}s]")
                            
                            # Sanity check: spike times should be non-negative and within reasonable duration
                            if spike_min < -0.01:  # Allow small negative values due to numerical precision
                                print(f"      ⚠ Warning: Negative spike times detected (min={spike_min:.6f}s)")
                    
                        # Check if any spikes fall within EEG time range
                        # Both should be in relative time [0, duration]
                        n_in_range = np.sum((spike_times >= eeg_time[0]) & 
                                           (spike_times <= eeg_time[-1]))
                        
                        if n_in_range == 0:
                            print(f"      ⚠ Warning: 0 spikes in EEG range [{eeg_time[0]:.2f}, {eeg_time[-1]:.2f}]s")
                            print(f"         EEG time format: {eeg_info.get('time_format', 'unknown')}")
                            print(f"         Spike time format: {time_format}")
                            continue
                        
                        # Perform correlation analysis
                        # No time alignment needed - both spike_times and eeg_time are in relative coordinates
                        correlation_results = self.analyze_channel_spike_correlation(
                            eeg_data, eeg_time, spike_times, channel, state
                        )
                        
                        if correlation_results is not None:
                            channel_spike_results[analysis_key] = correlation_results
                            print(f"    ✓ Completed {analysis_key}")
                
                if channel_spike_results:
                    results['movement_states'][state][channel] = channel_spike_results
                    print(f"  Completed {channel}: {len(channel_spike_results)} spike components analyzed")
        
        # Compute summary statistics
        print(f"\n{'-'*70}")
        print("Computing summary statistics...")
        results['summary_statistics'] = self.compute_summary_statistics(results)
        
        print(f"{'='*70}")
        print(f"Analysis complete for dataset: {dataset_name}")
        print(f"{'='*70}\n")
        
        return results
    
    def compute_summary_statistics(self, analysis_results: Dict) -> Dict[str, Any]:
        """
        Compute summary statistics across all analyses
        Aggregates STA and PPC values across states and channels
        
        Args:
            analysis_results: Complete analysis results dictionary
            
        Returns:
            Summary statistics dictionary with aggregated metrics
        """
        # Collect all unique frequency bands used
        all_bands = set()
        movement_states = analysis_results.get('movement_states', {})
        
        for state_data in movement_states.values():
            for channel_data in state_data.values():
                for analysis_data in channel_data.values():
                    bands = analysis_data.get('bands', {})
                    all_bands.update(bands.keys())
        
        summary = {
            'total_analyses': 0,
            'channels_analyzed': set(),
            'states_analyzed': set(),
            'frequency_bands': sorted(list(all_bands)),
            'ppc_statistics': {},
            'sta_statistics': {}
        }
        
        # Initialize band-specific summaries for all detected bands
        for band in all_bands:
            summary['ppc_statistics'][band] = []
            summary['sta_statistics'][band] = []
        
        # Collect statistics from all analyses
        for state_name, state_data in movement_states.items():
            summary['states_analyzed'].add(state_name)
            
            for channel_name, channel_data in state_data.items():
                summary['channels_analyzed'].add(channel_name)
                
                for analysis_key, analysis_data in channel_data.items():
                    summary['total_analyses'] += 1
                    
                    bands = analysis_data.get('bands', {})
                    for band_name, band_data in bands.items():
                        # Collect PPC values
                        ppc_val = band_data.get('ppc', {}).get('ppc', 0.0)
                        if ppc_val > 0:
                            summary['ppc_statistics'][band_name].append(ppc_val)
                        
                        # Collect STA peak values
                        sta_stats = band_data.get('sta', {})
                        if 'mean' in sta_stats and len(sta_stats['mean']) > 0:
                            peak_val = np.max(np.abs(sta_stats['mean']))
                            summary['sta_statistics'][band_name].append(peak_val)
        
        # Convert sets to lists for JSON serialization
        summary['channels_analyzed'] = sorted(list(summary['channels_analyzed']))
        summary['states_analyzed'] = sorted(list(summary['states_analyzed']))
        
        # Compute statistics for each measure and band
        for band in all_bands:
            for measure in ['ppc_statistics', 'sta_statistics']:
                values = summary[measure][band]
                if values:
                    summary[measure][band] = {
                        'values': values,
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_samples': len(values)
                    }
                else:
                    summary[measure][band] = {
                        'values': [],
                        'mean': 0.0, 'std': 0.0, 'median': 0.0, 
                        'min': 0.0, 'max': 0.0, 'n_samples': 0
                    }
        
        return summary
    
    def visualize_correlation_results(self, analysis_results: Dict, output_folder: str = None):
        """
        Create comprehensive visualizations of correlation analysis results
        Generates separate plots for STA and PPC organized by state, mode, and channel
    
        Args:
            analysis_results: Analysis results dictionary
            output_folder: Output directory for saving plots
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        dataset_name = analysis_results.get('dataset_name', 'unknown')
    
        # Generate STA and PPC plots
        print("\nGenerating visualizations...")
        self._plot_sta_by_mode_channel(analysis_results, output_folder)
        self._plot_ppc_by_mode_channel(analysis_results, output_folder)
    
        print(f"\n✓ Visualizations saved to {output_folder}")

    def _plot_sta_by_mode_channel(self, analysis_results: Dict, output_folder: str = None):
        """
        Plot STA results organized by movement state, mode, and EMG channel (separately)
        Each component shows all frequency bands in separate subplots
        """
        sns.set_style('whitegrid')
        movement_states = analysis_results.get('movement_states', {})

        # Define colors for frequency bands (支援 mu 和 alpha)
        band_colors = {
            'delta': '#1f77b4',      # Blue
            'theta': '#ff7f0e',      # Orange
            'mu': '#2ca02c',         # Green (motor channels)
            'alpha': '#2ca02c',      # Green (visual channels, same as mu)
            'beta': '#d62728',       # Red
            'low_gamma': '#9467bd'   # Purple
        }

        for state_name, state_data in movement_states.items():
            if not state_data:
                continue

            for eeg_channel_name, channel_data in state_data.items():
                # Determine channel group for title
                if eeg_channel_name in self.motor_channels:
                    channel_type = "Motor Control"
                elif eeg_channel_name in self.visual_channels:
                    channel_type = "Visual Processing"
                else:
                    channel_type = "Other"

                # Group data by mode and EMG channel
                components_by_mode_emg = {}
                for analysis_key, analysis_data in channel_data.items():
                    if '_' in analysis_key:
                        parts = analysis_key.split('_')
                        mode = parts[0]  # single or multi
                        emg_channel = parts[1] if len(parts) >= 3 else 'unknown'
                        comp_name = '_'.join(parts[2:]) if len(parts) >= 3 else analysis_key

                        mode_emg_key = f"{mode}_{emg_channel}"

                        if mode_emg_key not in components_by_mode_emg:
                            components_by_mode_emg[mode_emg_key] = {}

                        components_by_mode_emg[mode_emg_key][comp_name] = analysis_data

                # Create one plot per (mode, emg_channel) combination
                for mode_emg_key, components in components_by_mode_emg.items():
                    if not components:
                        continue

                    parts = mode_emg_key.split('_')
                    mode = parts[0]
                    emg_channel = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'

                    # Process each component separately (one figure per component)
                    for comp_name, analysis_data in components.items():
                        bands = analysis_data.get('bands', {})
                        if not bands:
                            continue

                        # Get number of frequency bands with data
                        n_bands = len(bands)
                        if n_bands == 0:
                            continue

                        # Create figure with subplots for each frequency band
                        fig, axes = plt.subplots(n_bands, 1, figsize=(14, 4 * n_bands))
                        if n_bands == 1:
                            axes = [axes]

                        has_data = False
                        band_idx = 0

                        # Plot each frequency band in separate subplot
                        for band_name, band_data in bands.items():
                            if band_idx >= len(axes):
                                break
                            
                            ax = axes[band_idx]
                            sta_stats = band_data.get('sta', {})
                            relative_time = band_data.get('relative_time', [])
                            freq_range = band_data.get('freq_range', (0, 0))

                            if 'mean' in sta_stats and len(sta_stats['mean']) > 0:
                                mean = sta_stats['mean']
                                sem = sta_stats.get('sem', np.zeros_like(mean))
                                n_spikes = sta_stats.get('n_spikes', 0)
                            
                                color = band_colors.get(band_name, '#000000')
                            
                                # Plot mean line
                                ax.plot(relative_time, mean, '-', linewidth=2.5,
                                    color=color, alpha=0.8)

                                # Add SEM shaded area
                                ax.fill_between(relative_time, mean - sem, mean + sem,
                                            color=color, alpha=0.15)

                                # Add spike time marker and baseline reference
                                ax.axvline(0, color='black', linestyle='--', alpha=0.5, 
                                        linewidth=2, label='Spike time')
                                ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

                                # Set title with frequency band info
                                band_title = f'{band_name.title()} ({freq_range[0]}-{freq_range[1]} Hz)'
                                ax.set_title(f'{band_title} | n_spikes = {n_spikes}', 
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

                        # Overall title for the figure
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
                            # Create state-specific subfolder within STA folder
                            state_folder_name = state_name.replace("_", " ").title()
                            plot_save_folder = os.path.join(output_folder, eeg_channel_name, 
                                                        mode_title, 'STA', state_folder_name)
                            os.makedirs(plot_save_folder, exist_ok=True)
                            filename_base = f'sta_{emg_channel}_{comp_name}'
                            png_path = os.path.join(plot_save_folder, f'{filename_base}.png')
                            plt.savefig(png_path, dpi=300, bbox_inches='tight')
                            print(f"  Saved STA plot: {png_path}")

                        plt.close(fig)

    def _plot_ppc_by_mode_channel(self, analysis_results: Dict, output_folder: str = None):
        """
        PPC heatmap per (state, EEG channel, mode) combination
        Rows = frequency bands (including mu for motor, alpha for visual)
        Columns = all EMG components combined
        """
        sns.set_style('whitegrid')
        movement_states = analysis_results.get('movement_states', {})

        for state_name, state_data in movement_states.items():
            if not state_data:
                continue

            for eeg_channel_name, channel_data in state_data.items():
                # Determine channel group for title
                if eeg_channel_name in self.motor_channels:
                    channel_type = "Motor Control"
                elif eeg_channel_name in self.visual_channels:
                    channel_type = "Visual Processing"
                else:
                    channel_type = "Other"

                # Group data by mode
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

                # Create heatmap for each mode
                for mode, components in components_by_mode.items():
                    if not components:
                        continue

                    # Collect all unique bands from this channel's data
                    all_bands = set()
                    for analysis_data in components.values():
                        all_bands.update(analysis_data.get('bands', {}).keys())
                    
                    # Sort bands by frequency order (low to high) based on channel type
                    band_order = self._get_band_order_for_channel(eeg_channel_name)
                    bands = [b for b in band_order if b in all_bands]
                    component_names = list(components.keys())
                    
                    if len(component_names) == 0 or len(bands) == 0:
                        continue

                    # Prepare DataFrames
                    ppc_df = pd.DataFrame(index=bands, columns=component_names, dtype=float)
                    nsp_df = pd.DataFrame(index=bands, columns=component_names, dtype=int)
                    freq_ranges = {}

                    for comp_label, analysis_data in components.items():
                        band_dict = analysis_data.get('bands', {})
                        for band_name in bands:
                            if band_name in band_dict:
                                ppc_val = band_dict[band_name].get('ppc', {}).get('ppc', np.nan)
                                n_spikes = int(band_dict[band_name].get('ppc', {}).get('n_spikes', 0))
                                freq_range = band_dict[band_name].get('freq_range', (0, 0))
                                
                                ppc_df.at[band_name, comp_label] = ppc_val
                                nsp_df.at[band_name, comp_label] = n_spikes
                                freq_ranges[band_name] = freq_range
                            else:
                                ppc_df.at[band_name, comp_label] = np.nan
                                nsp_df.at[band_name, comp_label] = 0

                    if ppc_df.isna().all().all():
                        continue

                    # Sort columns
                    def sort_key(col_name):
                        parts = col_name.split('-')
                        emg_ch = parts[0] if len(parts) > 0 else ''
                        comp = parts[1] if len(parts) > 1 else ''
                        return (emg_ch, comp)
                    
                    col_order = sorted(component_names, key=sort_key)
                    ppc_df = ppc_df[col_order]
                    nsp_df = nsp_df[col_order]

                    # Build annotations
                    annot = ppc_df.copy().astype(object)
                    for r in ppc_df.index:
                        for c in ppc_df.columns:
                            val = ppc_df.at[r, c]
                            nsp = int(nsp_df.at[r, c])
                            if np.isnan(val):
                                annot.at[r, c] = ""
                            else:
                                annot.at[r, c] = f"{val:.3f}\n(n={nsp})"

                    # Plot heatmap
                    figsize = (max(12, len(component_names) * 0.8), 
                              max(5, len(bands) * 1.0))
                    fig, ax = plt.subplots(figsize=figsize)
                    sns.heatmap(ppc_df.astype(float), annot=annot, fmt='', 
                               cmap='viridis', vmin=0.0, vmax=1.0, 
                               linewidths=0.5, linecolor='white',
                               cbar_kws={'label': 'PPC (0-1)'}, ax=ax)

                    state_title = state_name.replace("_", " ").title()
                    mode_title = mode.title()
                    ax.set_title(
                        f'PPC Heatmap — All EMG Channels & Frequency Bands\n'
                        f'{state_title} | {mode_title} Mode | '
                        f'EEG: {eeg_channel_name} ({channel_type})',
                        fontsize=12, fontweight='bold'
                    )
                    ax.set_xlabel('EMG Channel - Spike Component', fontsize=11)
                    ax.set_ylabel('Frequency Band', fontsize=11)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, 
                                      ha='right', fontsize=9)
                    
                    # Format y-axis labels with frequency ranges
                    y_labels = []
                    for band_name in ppc_df.index:
                        freq_range = freq_ranges.get(band_name, (0, 0))
                        y_labels.append(f'{band_name.title()}\n({freq_range[0]}-{freq_range[1]} Hz)')
                    ax.set_yticklabels(y_labels, fontsize=9)
                    
                    # Add vertical separators between EMG channels
                    current_emg = None
                    for idx, col_name in enumerate(col_order):
                        emg_ch = col_name.split('-')[0]
                        if current_emg is not None and emg_ch != current_emg:
                            ax.axvline(idx, color='white', linewidth=2.5, alpha=0.8)
                        current_emg = emg_ch
                    
                    plt.tight_layout()

                    if output_folder:
                        plot_save_folder = os.path.join(output_folder, eeg_channel_name, 
                                                       mode_title, 'PPC')
                        os.makedirs(plot_save_folder, exist_ok=True)
                        filename = f'ppc_heatmap_{state_name}_all_channels.png'
                        png_path = os.path.join(plot_save_folder, filename)
                        plt.savefig(png_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved PPC heatmap: {png_path}")

                    plt.close(fig)

    def save_analysis_results(self, analysis_results: Dict, output_folder: str):
        """
        Save complete analysis results to files
        Saves both pickle (complete data) and CSV (summary)
        
        Args:
            analysis_results: Complete analysis results dictionary
            output_folder: Output directory
        """
        os.makedirs(output_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = analysis_results.get('dataset_name', 'unknown')
        
        # Save complete results as pickle
        pickle_path = os.path.join(output_folder, 
                                   f'correlation_analysis_{dataset_name}_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(analysis_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save summary as CSV (include both mu and alpha)
        summary = analysis_results.get('summary_statistics', {})
        summary_rows = []
        
        all_bands = summary.get('frequency_bands', [])
        for band_name in all_bands:
            for measure in ['ppc_statistics', 'sta_statistics']:
                stats = summary.get(measure, {}).get(band_name, {})
                if isinstance(stats, dict) and 'mean' in stats:
                    # Try to get frequency range from results
                    freq_range_str = "N/A"
                    
                    # Search through results to find frequency range for this band
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
                        'frequency_band': band_name,
                        'freq_range_Hz': freq_range_str,
                        'measure': measure.replace('_statistics', '').upper(),
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'median': stats['median'],
                        'min': stats['min'],
                        'max': stats['max'],
                        'n_samples': stats['n_samples']
                    })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            csv_path = os.path.join(output_folder, 
                                   f'correlation_summary_{dataset_name}_{timestamp}.csv')
            summary_df.to_csv(csv_path, index=False)
            print(f"\n✓ Results saved:")
            print(f"  - Complete data: {os.path.basename(pickle_path)}")
            print(f"  - Summary table: {os.path.basename(csv_path)}")
        else:
            print(f"\n✓ Results saved: {os.path.basename(pickle_path)}")


# ==================== Main Processing Functions ====================

def analyze_single_dataset_correlations(segments_file_path: str, 
                                       output_folder: str = None) -> Dict:
    """
    Analyze spike-EEG correlations for a single dataset
    
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
    analyzer = SpikeTriggeredAnalyzer()
    
    # Run correlation analysis
    results = analyzer.analyze_dataset_correlations(segments_data)
    
    # Determine output folder
    if output_folder is None:
        base_dir = os.path.dirname(segments_file_path)
        output_folder = os.path.join(base_dir, 'correlation_analysis')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Save results and create visualizations
    analyzer.save_analysis_results(results, output_folder)
    analyzer.visualize_correlation_results(results, output_folder)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_folder}\n")
    
    return results


def analyze_all_datasets_correlations(base_path: str, 
                                      dataset_prefix: str = 'path2_') -> Dict:
    """
    Analyze spike-EEG correlations for all datasets in base directory
    Processes multiple datasets and generates master summary
    
    Args:
        base_path: Base directory containing dataset folders
        dataset_prefix: Prefix for dataset folder names (e.g., 'path2_')
        
    Returns:
        Dictionary of results for all datasets
    """
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
        output_folder = os.path.join(dataset_path, 'correlation_analysis')
        
        try:
            results = analyze_single_dataset_correlations(segments_file, output_folder)
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
                                      f'correlation_analysis_master_summary_{timestamp}.pkl')
    
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
            
            output_path = os.path.join(single_dataset, 'correlation_analysis')
            results = analyze_single_dataset_correlations(segments_files[0], output_path)

            print("Single dataset analysis completed!")
        else:
            print("No segmented data found for path2_02")

    # Example 2: Process all datasets (uncomment to use)
    # print(f"\n{'='*80}")
    # print("PROCESSING ALL DATASETS")
    # print(f"{'='*80}")
    # all_results = analyze_all_datasets_correlations(BASE_PATH, dataset_prefix='path2_')
    # print("\n✓ All datasets processing completed!")

# %%