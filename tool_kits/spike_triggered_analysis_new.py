# %%
"""
Spike-Triggered Analysis Module for EEG-EMG Correlation (Broadband, Single-Unit)

Performs broadband spike-triggered average (STA) analysis between EMG-derived spikes 
and EEG signals across movement states.

Key Features:
- Broadband STA: Average EEG response locked to spike times (no frequency filtering)
- Single-unit spikes only (multi-unit spikes excluded)
- Grid visualization: One figure per state×EMG, subplots for each EEG channel
- Multiple spike components shown as different lines within each subplot
- Works with relative time coordinates (0-based) from segmented data

Analysis Parameters:
- Extraction Window: ±500ms (captures sufficient EEG context around spikes)
- Display Window: ±500ms (uniform visualization window)

Target EEG Channels:
- Motor: Cz, C3, C4, FCz, FC3, FC4, CPz, CP3, CP4
- Visual: O1, O2, Oz

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
    """Spike-triggered analysis for EEG-EMG correlation: broadband STA with single-unit spikes"""
    
    def __init__(self, eeg_fs: float = 500.0, emg_fs: float = 1000.0):
        self.eeg_fs = eeg_fs
        self.emg_fs = emg_fs
        
        # Target EEG channels
        self.motor_channels = ['Cz', 'C3', 'C4', 'FCz', 'FC3', 'FC4', 'CPz', 'CP3', 'CP4']
        self.visual_channels = ['O1', 'O2', 'Oz']
        self.target_channels = self.motor_channels + self.visual_channels
        self.channel_groups = {'motor': self.motor_channels, 'visual': self.visual_channels}
        
        # Movement states
        self.movement_states = ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']
        
        # Analysis windows (broadband)
        self.extraction_window = 0.5  # ±500ms for segment extraction
        self.display_window = 0.5     # ±500ms for visualization
    
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
        """Reorganize spike data to {state: {comp_key: spike_data}} (single-unit only)"""
        reorganized_spikes = {state: {} for state in self.movement_states}
        spike_segments = segments_data.get('spike_segments', {})
        
        # Only process single-unit mode
        if 'single' not in spike_segments:
            return reorganized_spikes
        
        mode_data = spike_segments['single']
        
        for ch_label, ch_segments in mode_data.items():
            for comp_name, comp_segments in ch_segments.items():
                comp_key = f"{ch_label}_{comp_name}"
                for state in self.movement_states:
                    if state in comp_segments and comp_segments[state] is not None:
                        segment = comp_segments[state]
                        reorganized_spikes[state][comp_key] = {
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

    def extract_spike_triggered_segments(self, eeg_signal: np.ndarray, spike_times: np.ndarray,
                                       eeg_time: np.ndarray, window_size: float, 
                                       display_window: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract EEG segments around spike times (relative coordinates)"""
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
        """Compute STA: mean, std, n_spikes"""
        if eeg_segments.size == 0:
            return {'mean': np.array([]), 'std': np.array([]), 'n_spikes': 0}
        
        n_spikes = eeg_segments.shape[0]
        sta_mean = np.mean(eeg_segments, axis=0)
        sta_std = np.std(eeg_segments, axis=0)
        
        return {'mean': sta_mean, 'std': sta_std, 'n_spikes': n_spikes}
    
    def analyze_channel_spike_correlation(self, eeg_data: np.ndarray, eeg_time: np.ndarray,
                                    spike_times: np.ndarray, channel_name: str,
                                    movement_state: str) -> Dict[str, Any]:
        """STA analysis for single EEG channel (broadband, no filtering)"""
        channel_type = 'Visual' if channel_name in self.visual_channels else 'Motor'
        
        results = {
            'channel': channel_name,
            'channel_type': channel_type,
            'movement_state': movement_state,
            'n_spikes': len(spike_times),
            'sta': None
        }
        
        # Extract segments using fixed window (±500ms)
        eeg_segments, relative_time, display_time = self.extract_spike_triggered_segments(
            eeg_data, spike_times, eeg_time, self.extraction_window, self.display_window
        )
        
        # Compute STA
        if len(eeg_segments) > 0:
            results['sta'] = self.compute_spike_triggered_average(eeg_segments)
            results['relative_time'] = relative_time
            results['display_time'] = display_time
            results['extraction_window'] = self.extraction_window
            results['display_window'] = self.display_window
        
        return results if results['sta'] is not None else None
    
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
                
                # Only process single-unit spikes
                state_spikes = reorganized_spikes.get(state, {})
                
                for comp_name, comp_data in state_spikes.items():
                    if 'spike_times' not in comp_data or len(comp_data['spike_times']) < 2:
                        continue
                    
                    analysis_key = f"single_{comp_name}"
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
        """Aggregate STA statistics across all analyses"""
        movement_states = analysis_results.get('movement_states', {})
        
        summary = {
            'total_analyses': 0,
            'channels_analyzed': set(),
            'states_analyzed': set(),
            'sta_statistics': []
        }
        
        for state_name, state_data in movement_states.items():
            summary['states_analyzed'].add(state_name)
            for channel_name, channel_data in state_data.items():
                summary['channels_analyzed'].add(channel_name)
                for analysis_key, analysis_data in channel_data.items():
                    summary['total_analyses'] += 1
                    sta_stats = analysis_data.get('sta', {})
                    if sta_stats is not None and 'mean' in sta_stats and len(sta_stats['mean']) > 0:
                        summary['sta_statistics'].append(np.max(np.abs(sta_stats['mean'])))
        
        summary['channels_analyzed'] = sorted(list(summary['channels_analyzed']))
        summary['states_analyzed'] = sorted(list(summary['states_analyzed']))
        
        # Compute aggregate statistics
        values = summary['sta_statistics']
        summary['sta_statistics'] = {
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
        """Generate STA grid visualizations organized by state and EMG channel"""
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        print("\nGenerating visualizations...")
        self._plot_sta_grid(analysis_results, output_folder)
        
        if output_folder:
            print(f"\n✓ All visualizations saved to: {output_folder}")

    def _plot_sta_grid(self, analysis_results: Dict, output_folder: str = None):
        """Plot STA grid: one figure per state×EMG, subplots for EEG channels"""
        sns.set_style('whitegrid')
        movement_states = analysis_results.get('movement_states', {})
        
        # Component colors for visual distinction
        component_colors = plt.cm.tab10.colors
        
        for state_name, state_data in movement_states.items():
            if not state_data:
                continue
            
            # Group by EMG channel
            data_by_emg = {}  # {emg_channel: {eeg_ch: {comp_key: sta_data}}}
            
            # Reorganize data structure
            for eeg_channel_name, channel_data in state_data.items():
                for analysis_key, analysis_data in channel_data.items():
                    # Extract EMG channel and component from analysis_key
                    # analysis_key format: "single_EMG_L_comp1"
                    parts = analysis_key.split('_')
                    if len(parts) < 3:
                        continue
                    
                    emg_channel = parts[1]  # "EMG" or full EMG channel like "EMG_L"
                    # Handle cases where EMG channel might have underscore (EMG_L, EMG_R)
                    if len(parts) > 3 and parts[2] in ['L', 'R']:
                        emg_channel = f"{parts[1]}_{parts[2]}"
                        comp_name = '_'.join(parts[3:])
                    else:
                        comp_name = '_'.join(parts[2:])
                    
                    if emg_channel not in data_by_emg:
                        data_by_emg[emg_channel] = {}
                    if eeg_channel_name not in data_by_emg[emg_channel]:
                        data_by_emg[emg_channel][eeg_channel_name] = {}
                    
                    data_by_emg[emg_channel][eeg_channel_name][comp_name] = analysis_data
            
            # Create one figure per EMG channel
            for emg_channel, eeg_data in data_by_emg.items():
                if not eeg_data:
                    continue
                
                n_eeg_channels = len(eeg_data)
                n_cols = 3
                n_rows = int(np.ceil(n_eeg_channels / n_cols))
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                if n_rows * n_cols == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                
                for idx, (eeg_ch, comp_dict) in enumerate(sorted(eeg_data.items())):
                    if idx >= len(axes):
                        break
                    
                    ax = axes[idx]
                    
                    # Plot each component as a separate line
                    for comp_idx, (comp_name, sta_data) in enumerate(sorted(comp_dict.items())):
                        sta_results = sta_data.get('sta')
                        time_axis = sta_data.get('display_time', [])
                        
                        if sta_results and 'mean' in sta_results and len(sta_results['mean']) > 0:
                            mean = sta_results['mean']
                            n_spikes = sta_results.get('n_spikes', 0)
                            
                            # Use different colors for different components
                            color = component_colors[comp_idx % len(component_colors)]
                            
                            # Plot with label
                            label = f"{comp_name} (n={n_spikes})"
                            ax.plot(time_axis, mean, label=label, linewidth=2, color=color)
                    
                    # Styling
                    ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
                    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                    ax.set_title(eeg_ch, fontweight='bold', fontsize=11)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    ax.set_ylabel('EEG (µV)', fontsize=9)
                    ax.legend(fontsize=7, loc='best', framealpha=0.9)
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for idx in range(n_eeg_channels, len(axes)):
                    axes[idx].axis('off')
                
                # Overall title
                state_title = state_name.replace("_", " ").title()
                plt.suptitle(f'Spike-Triggered Average \n{state_title} | EMG: {emg_channel}', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save figure
                if output_folder:
                    # Create folder structure: state/STA/
                    state_folder = state_name.replace('_', ' ').title().replace(' ', '_')
                    save_folder = os.path.join(output_folder, state_folder, 'STA')
                    os.makedirs(save_folder, exist_ok=True)
                    
                    filename = f'sta_grid_{emg_channel}.png'
                    filepath = os.path.join(save_folder, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"  Saved: {state_folder}/STA/{filename}")
                plt.close(fig)

    def save_analysis_results(self, analysis_results: Dict, output_folder: str):
        """Save analysis results to pickle (complete) and CSV (summary)"""
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = analysis_results.get('dataset_name', 'unknown')
        
        # Save complete results as pickle
        pickle_path = os.path.join(output_folder, f'correlation_analysis_{dataset_name}_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(analysis_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save summary as CSV
        summary = analysis_results.get('summary_statistics', {})
        summary_rows = []
        
        sta_stats = summary.get('sta_statistics', {})
        if isinstance(sta_stats, dict) and 'mean' in sta_stats:
            summary_rows.append({
                'measure': 'STA',
                'mean': sta_stats['mean'],
                'std': sta_stats['std'],
                'median': sta_stats['median'],
                'min': sta_stats['min'],
                'max': sta_stats['max'],
                'n_samples': sta_stats['n_samples']
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
    BASE_PATH = os.environ["RESEARCH_DATA_ROOT"]
    
    # Single dataset example
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
    
    # Batch processing example (uncomment to use)
    # all_results = analyze_all_datasets_correlations(BASE_PATH, dataset_prefix='path2_')
# %%

