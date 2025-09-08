# %%
"""
This module provides unified segmentation and visualization of EEG, EMG, and spike data
according to movement states (e.g., sit-to-stand, steady gait) for multi-modal neural datasets.
Key features:
- Loads movement segmentation results, spike detection results, and EMG/EEG data for each dataset.
- Segments temporal data (EEG, EMG, CWF components, spikes) by movement state with precise time alignment.
- Resamples EMG-derived CWF components to EEG sampling rate for cross-modal analysis.
- Generates spike trains for each movement segment and computes summary statistics.
- Saves segmented data and summary tables, and provides visualization functions for all modalities.
- Supports batch processing of multiple datasets and robust error handling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pickle
from typing import Dict, Optional, Tuple, Union, Any
from scipy.interpolate import interp1d

class MovementDataSegmentor:
    """Segmentation of EEG/EMG data by movement states with spike support and CWF resampling"""
    
    def __init__(self):
        self.states = ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']
        self.eeg_fs = 500  # EEG sampling rate
        self.emg_fs = 1000  # EMG sampling rate
    
    def load_movement_results(self, dataset_path: str) -> Optional[Dict]:
        """Load movement segmentation results from pkl file"""
        pattern = os.path.join(dataset_path, 'movement_analysis', 'movement_results_*.pkl')
        files = glob.glob(pattern)
        if not files:
            print(f"No movement results found in {dataset_path}")
            return None
        
        with open(files[0], 'rb') as f:
            return pickle.load(f)
    
    def load_spike_data(self, dataset_path: str) -> Dict:
        """Load spike detection results from firing_pattern_to_spikes"""
        spike_pattern = os.path.join(dataset_path, 'firing_pattern_to_spikes', 'spike_results_*.pkl')
        files = glob.glob(spike_pattern)
        if not files:
            print(f"No spike results found in {dataset_path}")
            return {}
        
        with open(files[0], 'rb') as f:
            return pickle.load(f)
    
    def load_emg_timestamps(self, dataset_path: str) -> Optional[np.ndarray]:
        """Load EMG timestamps from multi-kernel-deconv results"""
        for mode in ['single', 'multi']:
            ts_path = os.path.join(dataset_path, 'multi-kernel-deconv', mode, 'emg_timestamps.npy')
            if os.path.exists(ts_path):
                return np.load(ts_path)
        
        print(f"No emg_timestamps.npy found in {dataset_path}")
        return None
    
    def load_eeg_data(self, dataset_path: str) -> Dict:
        """Load EEG CSV files from EEG folder"""
        eeg_folder = os.path.join(dataset_path, 'EEG')
        eeg_data = {}
        
        if not os.path.exists(eeg_folder):
            return eeg_data
        
        for csv_file in glob.glob(os.path.join(eeg_folder, '*.csv')):
            try:
                df = pd.read_csv(csv_file)
                if df.shape[1] >= 2:  # At least time + 1 channel
                    basename = os.path.splitext(os.path.basename(csv_file))[0]
                    eeg_data[basename] = {
                        'time': df.iloc[:, 0].values,
                        'data': df.iloc[:, 1:].values,
                        'channels': df.columns[1:].tolist(),
                        'fs': self.eeg_fs
                    }
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return eeg_data
    
    def resample_cwf_to_eeg_fs(self, cwf_data: np.ndarray, emg_time: np.ndarray, 
                              target_fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample CWF_comp from EMG fs to EEG fs with time preservation
        """
        if len(cwf_data) != len(emg_time):
            raise ValueError("CWF data length must match EMG time length")
        
        if len(emg_time) < 2:
            raise ValueError("Need at least 2 time points for resampling")
        
        original_fs = 1.0 / np.mean(np.diff(emg_time))
        
        if abs(original_fs - target_fs) < 1:
            return cwf_data, emg_time
        
        # Create new time vector at target sampling rate
        start_time = emg_time[0]
        end_time = emg_time[-1]
        duration = end_time - start_time
        n_samples_new = int(duration * target_fs) + 1
        resampled_time = np.linspace(start_time, end_time, n_samples_new)
        
        # Interpolate CWF data
        if cwf_data.ndim == 1:
            f = interp1d(emg_time, cwf_data, kind='linear', bounds_error=False, fill_value='extrapolate')
            resampled_cwf = f(resampled_time)
        else:
            resampled_cwf = np.zeros((len(resampled_time), cwf_data.shape[1]))
            for i in range(cwf_data.shape[1]):
                f = interp1d(emg_time, cwf_data[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
                resampled_cwf[:, i] = f(resampled_time)
        
        print(f"  Resampled CWF: {cwf_data.shape} -> {resampled_cwf.shape} (fs: {original_fs:.1f} -> {target_fs:.1f} Hz)")
        return resampled_cwf, resampled_time
    
    def load_npz_data(self, dataset_path: str) -> Dict:
        """Load NPZ deconvolution results with CWF resampling"""
        npz_data = {}
        
        emg_timestamps = self.load_emg_timestamps(dataset_path)
        if emg_timestamps is None:
            print("Warning: No EMG timestamps found, cannot resample CWF_comp")
            return npz_data
        
        for mode in ['single', 'multi']:
            mode_dir = os.path.join(dataset_path, 'multi-kernel-deconv', mode)
            if not os.path.exists(mode_dir):
                continue
                
            pattern = os.path.join(mode_dir, 'multi_kernel_deconv_results_*.npz')
            npz_files = glob.glob(pattern)
            if not npz_files:
                continue
            
            npz_file = sorted(npz_files)[-1]
            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    mode_data = {}
                    
                    channel_labels = ['LGA', 'RGA', 'LTA', 'RTA', 'LBF', 'RBF', 'LRF', 'RRF']
                    
                    for ch_label in channel_labels:
                        ch_data = {}
                        
                        # Only load and resample CWF_comp (remove firing pattern)
                        cwf_comp_key = f'{ch_label}_CWF_comp'
                        if cwf_comp_key in data:
                            cwf_comp_raw = data[cwf_comp_key]
                            
                            try:
                                cwf_comp_resampled, cwf_time_resampled = self.resample_cwf_to_eeg_fs(
                                    cwf_comp_raw, emg_timestamps, self.eeg_fs
                                )
                                
                                ch_data['CWF_comp'] = cwf_comp_resampled
                                ch_data['CWF_comp_time'] = cwf_time_resampled
                                ch_data['CWF_comp_fs'] = self.eeg_fs
                                
                                print(f"  Resampled {ch_label} CWF_comp: {cwf_comp_raw.shape} -> {cwf_comp_resampled.shape}")
                            except Exception as e:
                                print(f"  Error resampling {ch_label} CWF_comp: {e}")
                                ch_data['CWF_comp'] = cwf_comp_raw
                                ch_data['CWF_comp_time'] = emg_timestamps
                                ch_data['CWF_comp_fs'] = self.emg_fs
                        
                        if ch_data:
                            mode_data[ch_label] = ch_data
                    
                    npz_data[mode] = mode_data
                    
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
        
        return npz_data
    
    def find_time_indices(self, target_start: float, target_end: float, 
                         time_array: np.ndarray) -> np.ndarray:
        """Find all indices within time range using logical indexing"""
        time_mask = (time_array >= target_start) & (time_array <= target_end)
        indices = np.where(time_mask)[0]
        return indices
    
    def create_spike_train_for_segment(self, spike_times: np.ndarray, 
                                     segment_start: float, segment_end: float,
                                     target_fs: float) -> Dict:
        """Create spike train for a specific movement segment"""
        segment_spikes = spike_times[(spike_times >= segment_start) & (spike_times <= segment_end)]
        
        if len(segment_spikes) == 0:
            duration = segment_end - segment_start
            n_samples = int(duration * target_fs) + 1
            return {
                'spike_times': np.array([]),
                'spike_indices': np.array([]),
                'binary_train': np.zeros(n_samples, dtype=int),
                'time': np.linspace(segment_start, segment_end, n_samples),
                'n_spikes': 0,
                'fs': target_fs
            }
        
        duration = segment_end - segment_start
        n_samples = int(duration * target_fs) + 1
        time_vector = np.linspace(segment_start, segment_end, n_samples)
        
        binary_train = np.zeros(n_samples, dtype=int)
        spike_indices = []
        
        for spike_time in segment_spikes:
            closest_idx = np.argmin(np.abs(time_vector - spike_time))
            if closest_idx < len(binary_train):
                binary_train[closest_idx] = 1
                spike_indices.append(closest_idx)
        
        return {
            'spike_times': segment_spikes,
            'spike_indices': np.array(spike_indices),
            'binary_train': binary_train,
            'time': time_vector,
            'n_spikes': len(segment_spikes),
            'fs': target_fs
        }
    
    def get_absolute_movement_times(self, movement_results: Dict, data_start_time: float) -> Dict:
        """Convert relative movement times to absolute times based on data start time"""
        absolute_times = {}
        for state in self.states:
            if state in movement_results and movement_results[state] is not None:
                state_info = movement_results[state]
                absolute_times[state] = {
                    'start_time': state_info['start_time'] + data_start_time,
                    'end_time': state_info['end_time'] + data_start_time,
                    'duration': state_info['end_time'] - state_info['start_time']
                }
            else:
                absolute_times[state] = None
        return absolute_times
    
    def segment_temporal_data(self, data: np.ndarray, time_array: np.ndarray,
                             movement_results: Dict, data_start_time: float,
                             data_type: str, fs: float) -> Dict:
        """
        Unified function to segment temporal data (EEG/EMG) by movement states
        
        Args:
            data: Data array to segment
            time_array: Time array corresponding to data
            movement_results: Movement segmentation results
            data_start_time: Start time of the data recording
            data_type: Type identifier for the data
            fs: Sampling frequency
        """
        segments = {}
        absolute_times = self.get_absolute_movement_times(movement_results, data_start_time)
        
        for state in self.states:
            if absolute_times[state] is not None:
                abs_times = absolute_times[state]
                movement_start = abs_times['start_time']
                movement_end = abs_times['end_time']
                
                # Find time indices within movement period
                time_indices = self.find_time_indices(movement_start, movement_end, time_array)
                
                if len(time_indices) > 0:
                    start_idx = time_indices[0]
                    end_idx = time_indices[-1]
                    
                    segments[state] = {
                        'data': data[time_indices],
                        'time': time_array[time_indices],
                        'start_idx': int(start_idx),
                        'end_idx': int(end_idx),
                        'duration': abs_times['duration'],
                        'data_type': data_type,
                        'n_points': len(time_indices),
                        'movement_time': [movement_results[state]['start_time'], movement_results[state]['end_time']],
                        'aligned_time': [movement_start, movement_end],
                        'data_start_time': data_start_time,
                        'fs': fs
                    }
                    
                    print(f"  {state}: {data_type} -> indices[{start_idx}:{end_idx}] "
                          f"({len(time_indices)} points), "
                          f"time[{time_array[time_indices][0]:.2f}:{time_array[time_indices][-1]:.2f}]s")
                else:
                    print(f"  {state}: {data_type} no data points in time range "
                          f"[{movement_start:.2f}:{movement_end:.2f}]s")
                    segments[state] = None
            else:
                segments[state] = None
        
        return segments
    
    def segment_spike_data(self, spike_data: Dict, emg_timestamps: np.ndarray,
                          movement_results: Dict) -> Dict:
        """Segment spike data according to movement states using unified approach"""
        spike_segments = {}
        
        emg_start_time = emg_timestamps[0] if len(emg_timestamps) > 0 else 0.0
        absolute_times = self.get_absolute_movement_times(movement_results, emg_start_time)
        
        for mode, mode_data in spike_data.items():
            mode_segments = {}
            
            for ch_label, ch_data in mode_data.items():
                ch_segments = {}
                
                for comp_name, comp_data in ch_data.get('components', {}).items():
                    spike_times_relative = comp_data.get('spike_times_s', np.array([]))
                    
                    if len(spike_times_relative) == 0:
                        continue
                    
                    # Convert to absolute time
                    spike_times_absolute = spike_times_relative + emg_start_time
                    
                    comp_segments = {}
                    for state in self.states:
                        if absolute_times[state] is not None:
                            abs_times = absolute_times[state]
                            movement_start = abs_times['start_time']
                            movement_end = abs_times['end_time']
                            
                            spike_train_data = self.create_spike_train_for_segment(
                                spike_times_absolute, movement_start, movement_end, self.emg_fs
                            )
                            
                            if spike_train_data['n_spikes'] > 0:
                                comp_segments[state] = {
                                    'spike_times': spike_train_data['spike_times'],
                                    'spike_indices': spike_train_data['spike_indices'],
                                    'binary_train': spike_train_data['binary_train'],
                                    'time': spike_train_data['time'],
                                    'n_spikes': spike_train_data['n_spikes'],
                                    'data_type': f'Spikes-{ch_label}-{comp_name}',
                                    'movement_start': movement_start,
                                    'movement_end': movement_end,
                                    'duration': abs_times['duration'],
                                    'fs': spike_train_data['fs']
                                }
                                
                                print(f"  {state}: {ch_label}-{comp_name} -> {spike_train_data['n_spikes']} spikes "
                                      f"in [{movement_start:.2f}:{movement_end:.2f}]s")
                            else:
                                comp_segments[state] = None
                        else:
                            comp_segments[state] = None
                    
                    if comp_segments:
                        ch_segments[comp_name] = comp_segments
                
                if ch_segments:
                    mode_segments[ch_label] = ch_segments
            
            if mode_segments:
                spike_segments[mode] = mode_segments
        
        return spike_segments
    
    def segment_dataset(self, dataset_path: str) -> Dict:
        """Segment all data for one dataset with spike support and CWF resampling"""
        dataset_name = os.path.basename(dataset_path)
        print(f"Segmenting dataset: {dataset_name}")
        
        # Load movement segmentation results
        movement_results = self.load_movement_results(dataset_path)
        if movement_results is None:
            return {'error': 'No movement results found'}
        
        # Load EMG timestamps
        emg_timestamps = self.load_emg_timestamps(dataset_path)
        if emg_timestamps is None:
            print("Warning: No EMG timestamps found")
            return {'error': 'No EMG timestamps found'}
        
        emg_start_time = emg_timestamps[0]
        print(f"EMG start time: {emg_start_time:.3f}s")
        
        # Load spike data
        spike_data = self.load_spike_data(dataset_path)
        
        result = {
            'dataset_name': dataset_name,
            'movement_states': movement_results,
            'emg_start_time': emg_start_time,
            'eeg_segments': {},
            'emg_segments': {},
            'spike_segments': {},
            'available_states': []
        }
        
        # Determine available movement states
        for state in self.states:
            if state in movement_results and movement_results[state] is not None:
                result['available_states'].append(state)
        
        print(f"Available states: {result['available_states']}")
        
        # Segment EEG data using unified function
        eeg_data = self.load_eeg_data(dataset_path)
        for eeg_name, eeg_info in eeg_data.items():
            print(f"Segmenting EEG: {eeg_name}")
            eeg_start_time = eeg_info['time'][0]
            eeg_segments = self.segment_temporal_data(
                eeg_info['data'], eeg_info['time'], movement_results, 
                eeg_start_time, f"EEG-{eeg_name}", eeg_info['fs']
            )
            
            # Add EEG-specific metadata
            for state, segment in eeg_segments.items():
                if segment is not None:
                    segment['channels'] = eeg_info['channels']
            
            result['eeg_segments'][eeg_name] = eeg_segments
        
        # Segment EMG data using unified function
        npz_data = self.load_npz_data(dataset_path)
        for mode, mode_data in npz_data.items():
            print(f"Segmenting EMG mode: {mode}")
            mode_segments = {}
            
            for ch_label, ch_data in mode_data.items():
                if not isinstance(ch_data, dict):
                    continue
                
                ch_segments = {}
                
                # Segment resampled CWF_comp
                if 'CWF_comp' in ch_data and 'CWF_comp_time' in ch_data:
                    cwf_comp_data = ch_data['CWF_comp']
                    cwf_comp_time = ch_data['CWF_comp_time']
                    
                    segments = self.segment_temporal_data(
                        cwf_comp_data, cwf_comp_time, movement_results, 
                        emg_start_time, f"CWF_comp-{ch_label}-{mode}", self.eeg_fs
                    )
                    
                    # Add EMG-specific metadata
                    for state, segment in segments.items():
                        if segment is not None:
                            segment['resampled'] = True
                    
                    ch_segments['CWF_comp'] = segments
                
                if ch_segments:
                    mode_segments[ch_label] = ch_segments
            
            if mode_segments:
                result['emg_segments'][mode] = mode_segments
        
        # Segment spike data using specialized function
        if spike_data:
            print("Segmenting spike data")
            spike_segments = self.segment_spike_data(
                spike_data, emg_timestamps, movement_results
            )
            result['spike_segments'] = spike_segments
        
        return result
    
    def save_segments(self, segments: Dict, output_folder: str):
        """Save segmented data including spike segments"""
        os.makedirs(output_folder, exist_ok=True)
        dataset_name = segments['dataset_name']
        
        # Save complete segments
        segments_path = os.path.join(output_folder, f'segments_with_spikes_{dataset_name}.pkl')
        with open(segments_path, 'wb') as f:
            pickle.dump(segments, f)
        print(f"Segments with spikes saved: {segments_path}")
        
        # Create detailed summary CSV
        summary_data = []
        
        # EEG summary
        for eeg_name, eeg_segments in segments['eeg_segments'].items():
            for state in self.states:
                row = self._create_summary_row(segments, 'EEG', eeg_name, state, eeg_segments)
                summary_data.append(row)
        
        # EMG summary (CWF_comp resampled)
        for mode, mode_segments in segments['emg_segments'].items():
            for ch_label, ch_segments in mode_segments.items():
                if 'CWF_comp' in ch_segments:
                    data_segments = ch_segments['CWF_comp']
                    for state in self.states:
                        row = self._create_summary_row(
                            segments, f'EMG_{mode}_CWF_comp', ch_label, state, data_segments
                        )
                        summary_data.append(row)
        
        # Spike summary
        for mode, mode_segments in segments.get('spike_segments', {}).items():
            for ch_label, ch_segments in mode_segments.items():
                for comp_name, comp_segments in ch_segments.items():
                    for state in self.states:
                        row = self._create_spike_summary_row(
                            segments, f'Spike_{mode}', f'{ch_label}_{comp_name}', state, comp_segments
                        )
                        summary_data.append(row)
        
        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(output_folder, f'segmentation_summary_with_spikes_{dataset_name}.csv')
            summary_df.to_csv(csv_path, index=False)
            print(f"Summary with spikes saved: {csv_path}")
    
    def _create_summary_row(self, segments: Dict, data_type: str, channel: str, 
                           state: str, data_segments: Dict) -> Dict:
        """Helper to create summary row for regular data"""
        row = {
            'dataset': segments['dataset_name'],
            'data_type': data_type,
            'channel/file': channel,
            'state': state,
            'available': state in segments['available_states'],
            'has_data': state in data_segments and data_segments[state] is not None
        }
        
        if row['has_data']:
            segment = data_segments[state]
            row.update({
                'n_samples': len(segment['data']),
                'duration_s': segment['duration'],
                'start_time_s': segment['time'][0] if len(segment['time']) > 0 else None,
                'end_time_s': segment['time'][-1] if len(segment['time']) > 0 else None,
                'fs': segment.get('fs', 'unknown'),
                'resampled': segment.get('resampled', False),
                'data_start_time': segment.get('data_start_time', 'unknown')
            })
        else:
            row.update({
                'n_samples': None, 'duration_s': None, 'start_time_s': None,
                'end_time_s': None, 'fs': None, 'resampled': None, 'data_start_time': None
            })
        
        return row
    
    def _create_spike_summary_row(self, segments: Dict, data_type: str, channel: str,
                                 state: str, comp_segments: Dict) -> Dict:
        """Helper to create summary row for spike data"""
        row = {
            'dataset': segments['dataset_name'],
            'data_type': data_type,
            'channel/file': channel,
            'state': state,
            'available': state in segments['available_states'],
            'has_data': state in comp_segments and comp_segments[state] is not None
        }
        
        if row['has_data']:
            segment = comp_segments[state]
            row.update({
                'n_samples': len(segment['time']),
                'n_spikes': segment['n_spikes'],
                'duration_s': segment['duration'],
                'start_time_s': segment['time'][0] if len(segment['time']) > 0 else None,
                'end_time_s': segment['time'][-1] if len(segment['time']) > 0 else None,
                'fs': segment.get('fs', 'unknown'),
                'spike_rate_hz': segment['n_spikes'] / segment['duration'] if segment['duration'] > 0 else 0,
                'data_start_time': segments.get('emg_start_time', 'unknown')
            })
        else:
            row.update({
                'n_samples': None, 'n_spikes': None, 'duration_s': None,
                'start_time_s': None, 'end_time_s': None, 'fs': None, 'spike_rate_hz': None,
                'data_start_time': None
            })
        
        return row

    # ------------------- Visualization Functions -------------------
    def visualize_segments(self, segments: Dict, output_folder: str = None):
        """Create visualization of all segmented data including spikes and CWF components"""
        dataset_name = segments['dataset_name']
        available_states = segments['available_states']
        
        if not available_states:
            print("No states available for visualization")
            return
        
        colors = {'sit_to_stand': 'blue', 'stand_to_walk': 'green', 
                 'steady_gait': 'cyan', 'walk_to_stop': 'magenta'}
        
        # Visualize EEG segments
        self._plot_eeg_segments(segments, colors, output_folder)
        
        # Visualize CWF components (each channel, all components as subplots)
        self._plot_cwf_components_by_channel(segments, colors, output_folder)
        
        # Visualize Spike components (each channel, all components as subplots)
        self._plot_spike_components_by_channel(segments, colors, output_folder)
        
        # Plot spike rate analysis
        self.plot_spike_rate_analysis(segments, output_folder)
    
    def _plot_eeg_segments(self, segments: Dict, colors: Dict, output_folder: str = None):
        """Plot EEG segments with time alignment"""
        dataset_name = segments['dataset_name']
        available_states = segments['available_states']
        
        for eeg_name, eeg_segments in segments['eeg_segments'].items():
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            
            for state in available_states:
                if state in eeg_segments and eeg_segments[state] is not None:
                    segment = eeg_segments[state]
                    time = segment['time']
                    data = segment['data']
                    
                    # Plot average of all EEG channels
                    if data.ndim > 1:
                        y_data = np.mean(data, axis=1)
                    else:
                        y_data = data
                    
                    ax.plot(time, y_data, color=colors[state], 
                           label=f'{state}', alpha=0.8, linewidth=1.5)
            
            ax.set_ylabel('EEG Amplitude (µV)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'EEG Segments: {eeg_name} - {dataset_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                save_path = os.path.join(output_folder, f'EEG_{eeg_name}_segments_{dataset_name}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"EEG plot saved: {save_path}")
                plt.close()
            
            plt.tight_layout()
    
    def _plot_cwf_components_by_channel(self, segments: Dict, colors: Dict, output_folder: str = None):
        """Plot CWF components: each channel = one figure, each component = one subplot"""
        dataset_name = segments['dataset_name']
        available_states = segments['available_states']
        
        for mode, mode_segments in segments['emg_segments'].items():
            for ch_label, ch_segments in mode_segments.items():
                if 'CWF_comp' not in ch_segments:
                    continue
                
                cwf_segments = ch_segments['CWF_comp']
                
                # Determine number of CWF components by checking first available state
                n_components = 0
                sample_data = None
                for state in available_states:
                    if state in cwf_segments and cwf_segments[state] is not None:
                        sample_data = cwf_segments[state]['data']
                        if sample_data.ndim > 1:
                            n_components = sample_data.shape[1]
                        else:
                            n_components = 1
                        break
                
                if n_components == 0:
                    print(f"No CWF components found for {ch_label}")
                    continue
                
                # Create subplots: one per component
                fig, axes = plt.subplots(n_components, 1, figsize=(15, 4 * n_components))
                if n_components == 1:
                    axes = [axes]
                
                # Plot each component in separate subplot
                for comp_idx in range(n_components):
                    ax = axes[comp_idx]
                    has_data = False
                    
                    for state in available_states:
                        if state in cwf_segments and cwf_segments[state] is not None:
                            segment = cwf_segments[state]
                            time = segment['time']
                            data = segment['data']
                            
                            # Extract component data
                            if data.ndim > 1 and comp_idx < data.shape[1]:
                                y_data = data[:, comp_idx]
                            elif data.ndim == 1 and comp_idx == 0:
                                y_data = data
                            else:
                                continue
                            
                            ax.plot(time, y_data, color=colors[state], 
                                   label=f'{state}', alpha=0.8, linewidth=1.5)
                            has_data = True
                    
                    if has_data:
                        ax.set_ylabel('CWF Amplitude (µV)')
                        ax.set_title(f'Component {comp_idx + 1}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.set_title(f'Component {comp_idx + 1} - No data')
                
                # Set x-label for bottom subplot
                axes[-1].set_xlabel('Time (s)')
                
                plt.suptitle(f'{ch_label} - CWF Components ({mode} mode, {self.eeg_fs}Hz) - {dataset_name}', 
                            fontsize=14, y=0.98)
                plt.tight_layout()
                
                if output_folder:
                    os.makedirs(output_folder, exist_ok=True)
                    save_path = os.path.join(output_folder, f'CWF_Components_{ch_label}_{mode}_{dataset_name}.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"CWF components plot saved: {save_path}")
                    plt.close()
                
                
    def _plot_spike_components_by_channel(self, segments: Dict, colors: Dict, output_folder: str = None):
        """Plot Spike components: each channel = one figure, each component = one subplot"""
        dataset_name = segments['dataset_name']
        available_states = segments['available_states']
        spike_segments = segments.get('spike_segments', {})
        if not spike_segments:
            print("No spike segments available for plotting")
            return

        for mode, mode_segments in spike_segments.items():
            for ch_label, ch_segments in mode_segments.items():
                component_names = list(ch_segments.keys())
                n_components = len(component_names)
                if n_components == 0:
                    print(f"No spike components found for {ch_label}")
                    continue

                fig, axes = plt.subplots(n_components, 1, figsize=(15, 4 * n_components))
                if n_components == 1:
                    axes = [axes]

                for comp_idx, comp_name in enumerate(component_names):
                    ax = axes[comp_idx]
                    comp_segments = ch_segments[comp_name]
                    has_data = False
                    max_spike_rate = 0

                    # Prepare legend labels and handles
                    legend_labels = []
                    legend_handles = []

                    for state in available_states:
                        if state in comp_segments and comp_segments[state] is not None:
                            segment = comp_segments[state]
                            spike_times = segment['spike_times']
                            n_spikes = segment['n_spikes']
                            duration = segment['duration']
                            spike_rate = n_spikes / duration if duration > 0 else 0
                            max_spike_rate = max(max_spike_rate, spike_rate)

                            # Plot spike raster as vertical lines
                            for spike_time in spike_times:
                                ax.axvline(spike_time, color=colors[state], alpha=0.7, linewidth=2)

                            # Add legend entry with spike count and rate
                            legend_labels.append(f"{state}: {n_spikes} spikes ({spike_rate:.1f} Hz)")
                            legend_handles.append(plt.Line2D([0], [0], color=colors[state], lw=2))
                            has_data = True

                    if has_data:
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_ylabel('Spike Events')
                        ax.set_title(f'{comp_name} (Max rate: {max_spike_rate:.1f} Hz)')
                        ax.grid(True, alpha=0.3)
                        # Show legend with spike count and rate
                        ax.legend(legend_handles, legend_labels, loc='upper right')
                    else:
                        ax.set_title(f'{comp_name} - No data')

                axes[-1].set_xlabel('Time (s)')
                plt.suptitle(f'{ch_label} - Spike Components ({mode} mode, {self.emg_fs}Hz) - {dataset_name}',
                             fontsize=14, y=0.98)
                plt.tight_layout()
                if output_folder:
                    os.makedirs(output_folder, exist_ok=True)
                    save_path = os.path.join(output_folder, f'Spike_Components_{ch_label}_{mode}_{dataset_name}.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Spike components plot saved: {save_path}")
                    plt.close()
    
    def plot_spike_rate_analysis(self, segments: Dict, output_folder: str = None):
        """Plot spike rate analysis across movement states"""
        dataset_name = segments['dataset_name']
        available_states = segments['available_states']
        spike_segments = segments.get('spike_segments', {})
        
        if not spike_segments or not available_states:
            print("No spike data available for rate analysis")
            return
        
        # Collect spike rates for each channel/component/state
        spike_rate_data = []
        
        for mode, mode_segments in spike_segments.items():
            for ch_label, ch_segments in mode_segments.items():
                for comp_name, comp_segments in ch_segments.items():
                    for state in available_states:
                        if state in comp_segments and comp_segments[state] is not None:
                            segment = comp_segments[state]
                            spike_rate = segment['n_spikes'] / segment['duration'] if segment['duration'] > 0 else 0
                            
                            spike_rate_data.append({
                                'mode': mode,
                                'channel': ch_label,
                                'component': comp_name,
                                'state': state,
                                'spike_rate': spike_rate,
                                'n_spikes': segment['n_spikes'],
                                'duration': segment['duration']
                            })
        
        if not spike_rate_data:
            print("No spike rate data to plot")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(spike_rate_data)
    
        for mode in df['mode'].unique():
            df_mode = df[df['mode'] == mode]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
            # Bar plot
            state_rates = df_mode.groupby('state')['spike_rate'].agg(['mean', 'std']).reset_index()
            x_pos = np.arange(len(state_rates))
            bars = ax1.bar(x_pos, state_rates['mean'], yerr=state_rates['std'], 
                        capsize=5, alpha=0.7, 
                        color=['blue', 'green', 'cyan', 'magenta'][:len(state_rates)])
            ax1.set_xlabel('Movement State')
            ax1.set_ylabel('Average Spike Rate (Hz)')
            ax1.set_title(f'Average Spike Rate by Movement State - {mode} mode - {dataset_name}')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(state_rates['state'], rotation=45)
            ax1.grid(True, alpha=0.3)
            for i, (bar, mean_val, std_val) in enumerate(zip(bars, state_rates['mean'], state_rates['std'])):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.1,
                        f'{mean_val:.1f}±{std_val:.1f}',
                        ha='center', va='bottom', fontsize=10)
        
            # Heatmap
            pivot_data = df_mode.pivot_table(values='spike_rate', 
                                    index=['channel', 'component'], 
                                    columns='state', 
                                    aggfunc='mean', 
                                    fill_value=0)
            if not pivot_data.empty:
                im = ax2.imshow(pivot_data.values, cmap='viridis', aspect='auto')
                ax2.set_xticks(range(len(pivot_data.columns)))
                ax2.set_xticklabels(pivot_data.columns, rotation=45)
                ax2.set_yticks(range(len(pivot_data.index)))
                ax2.set_yticklabels([f'{ch}_{comp}' for ch, comp in pivot_data.index], fontsize=9)
                ax2.set_xlabel('Movement State')
                ax2.set_ylabel('Channel_Component')
                ax2.set_title(f'Spike Rate Heatmap (Hz) - {mode} mode')
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Spike Rate (Hz)')
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        text = ax2.text(j, i, f'{pivot_data.iloc[i, j]:.1f}',
                                    ha="center", va="center", color="white", fontsize=8)
            plt.tight_layout()
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                save_path = os.path.join(output_folder, f'Spike_Rate_Analysis_{mode}_{dataset_name}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Spike rate analysis saved: {save_path}")
                plt.close()
        return df


# Usage functions
def segment_single_dataset(dataset_path: str, output_folder: str = None):
    """Segment single dataset with spike support and CWF resampling"""
    segmentor = MovementDataSegmentor()
    segments = segmentor.segment_dataset(dataset_path)
    
    if 'error' not in segments:
        if output_folder is None:
            output_folder = os.path.join(dataset_path, 'neural_movement_segmentation')
        
        segmentor.save_segments(segments, output_folder)
        segmentor.visualize_segments(segments, output_folder)
        return segments
    else:
        print(f"Failed: {segments['error']}")
        return None

def segment_all_datasets(base_path: str):
    """Segment all datasets with spike support and CWF resampling"""
    segmentor = MovementDataSegmentor()
    datasets = sorted([d for d in glob.glob(os.path.join(base_path, 'path2_*')) 
                      if os.path.isdir(d)])
    
    if not datasets:
        print(f"No path2_ datasets found in {base_path}")
        return {}
    
    all_results = {}
    
    for i, dataset_path in enumerate(datasets, 1):
        dataset_name = os.path.basename(dataset_path)
        print(f"\n--- Processing {i}/{len(datasets)}: {dataset_name} ---")
        
        try:
            segments = segmentor.segment_dataset(dataset_path)
            
            if 'error' not in segments:
                all_results[dataset_name] = segments
                
                output_folder = os.path.join(dataset_path, 'neural_movement_segmentation')
                segmentor.save_segments(segments, output_folder)
                segmentor.visualize_segments(segments, output_folder)
                
                print(f"✓ Successfully processed {dataset_name}")
                print(f"  Results saved in: {output_folder}")
            else:
                print(f"✗ Failed to process {dataset_name}: {segments['error']}")
        
        except Exception as e:
            print(f"✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results


if __name__ == "__main__":
    BASE_PATH = '/home/ntk/exo/ExoskeletonPowerAsistance/simulation/mocap_EMG_EEG_data/20241225'
    
    print("=== Optimized Movement Data Segmentation ===")
    print("- Unified temporal data segmentation for EEG/EMG")
    print("- CWF_comp resampled from 1000Hz to 500Hz for EEG correlation")
    print("- Spike trains maintain original sampling rate")
    print("- Improved time alignment and code organization")
    
    # Process only path2_02
    single_path = os.path.join(BASE_PATH, 'path2_02')
    if os.path.exists(single_path):
        result = segment_single_dataset(single_path)
        if result:
            print(f"\n✓ path2_02 processing successful!")
            print(f"- Available states: {result['available_states']}")
            print(f"- EEG segments: {len(result['eeg_segments'])} files")
            print(f"- EMG segments: {len(result['emg_segments'])} modes")
            print(f"- Spike segments: {len(result.get('spike_segments', {}))} modes")
    else:
        print(f"path2_02 not found in {BASE_PATH}")
    
    # Uncomment to process all datasets
    # all_results = segment_all_datasets(BASE_PATH)
# %%