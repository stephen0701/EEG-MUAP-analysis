# %%
"""
Segments EEG, EMG data according to movement states from transient analysis
Fixed: Proper alignment of firing patterns and CWF using emg_timestamps
Modified: Individual firing pattern plots for each channel
Output: Save results in individual dataset folders with time alignment
Time Alignment: Aligns movement segmentation times to EEG/EMG time axes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pickle
import warnings
from typing import Dict, Optional

class MovementDataSegmentor:
    """Segmentation of EEG/EMG data by movement states with proper time alignment"""
    
    def __init__(self):
        self.states = ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']
    
    def load_movement_results(self, dataset_path: str) -> Optional[Dict]:
        """Load movement segmentation results from pkl file"""
        pattern = os.path.join(dataset_path, 'movement_analysis', 'movement_results_*.pkl')
        files = glob.glob(pattern)
        if not files:
            print(f"No movement results found in {dataset_path}")
            return None
        
        with open(files[0], 'rb') as f:
            return pickle.load(f)
    
    def load_emg_timestamps(self, dataset_path: str) -> Optional[np.ndarray]:
        """Load EMG timestamps from multi-kernel-deconv results"""
        # Try both single and multi modes
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
                        'channels': df.columns[1:].tolist()
                    }
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return eeg_data
    
    def load_npz_data(self, dataset_path: str) -> Dict:
        """Load NPZ deconvolution results with proper channel parsing"""
        npz_data = {}
        
        for mode in ['single', 'multi']:
            mode_dir = os.path.join(dataset_path, 'multi-kernel-deconv', mode)
            if not os.path.exists(mode_dir):
                continue
                
            # Find NPZ file
            pattern = os.path.join(mode_dir, 'multi_kernel_deconv_results_*.npz')
            npz_files = glob.glob(pattern)
            if not npz_files:
                continue
            
            npz_file = sorted(npz_files)[-1]  # Latest file
            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    mode_data = {}
                    
                    # Parse channel-specific data from NPZ
                    channel_labels = ['LGA', 'RGA', 'LTA', 'RTA', 'LBF', 'RBF', 'LRF', 'RRF']
                    
                    for ch_label in channel_labels:
                        ch_data = {}
                        
                        # Look for firing pattern data (X matrix)
                        x_key = f'{ch_label}_X'
                        if x_key in data:
                            ch_data['firing_pattern'] = data[x_key]
                        
                        # Look for CWF data
                        cwf_key = f'{ch_label}_CWF'
                        if cwf_key in data:
                            ch_data['CWF'] = data[cwf_key]
                        
                        # Look for component-wise CWF
                        cwf_comp_key = f'{ch_label}_CWF_comp'
                        if cwf_comp_key in data:
                            ch_data['CWF_comp'] = data[cwf_comp_key]
                        
                        # Look for recon data
                        recon_key = f'{ch_label}_recon_total'
                        if recon_key in data:
                            ch_data['recon_total'] = data[recon_key]
                        
                        # Look for per-component reconstructions
                        recon_comp_key = f'{ch_label}_recon_components'
                        if recon_comp_key in data:
                            ch_data['recon_components'] = data[recon_comp_key]
                        
                        if ch_data:  # Only add if we found some data for this channel
                            mode_data[ch_label] = ch_data
                    
                    # Also store any global data
                    for key in data.keys():
                        if not any(key.startswith(ch) for ch in channel_labels):
                            if not key.startswith('__'):  # Skip metadata
                                mode_data[key] = data[key]
                    
                    npz_data[mode] = mode_data
                    
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
        
        return npz_data
    
    def find_time_indices(self, target_start: float, target_end: float, 
                         time_array: np.ndarray) -> np.ndarray:
        """Find all indices within time range using logical indexing (like MATLAB's find)"""
        # Use logical indexing to find all points within time range
        time_mask = (time_array >= target_start) & (time_array <= target_end)
        indices = np.where(time_mask)[0]
        return indices
    
    def segment_data_by_time(self, data: np.ndarray, time_array: np.ndarray,
                           movement_results: Dict, data_type: str = "unknown") -> Dict:
        """
        Segment data according to movement states using logical indexing with time alignment
        
        Time alignment process (similar to MATLAB):
        aligned_time = movement_time + data_start_time
        where data_start_time is like time_EEG{i}(1) or time_EMG{i}(1)
        """
        segments = {}
        
        # Get the start time of this data (similar to MATLAB's time_EEG{i}(1) or time_EMG{i}(1))
        data_start_time = time_array[0] if len(time_array) > 0 else 0.0
        
        for state in self.states:
            if state in movement_results and movement_results[state] is not None:
                state_info = movement_results[state]
                # Original movement segmentation times
                movement_start = state_info['start_time']
                movement_end = state_info['end_time']
                
                # TIME ALIGNMENT: Align movement times to this data's time axis
                aligned_start = movement_start + data_start_time
                aligned_end = movement_end + data_start_time
                
                # Find all indices within aligned time range using logical indexing
                time_indices = self.find_time_indices(aligned_start, aligned_end, time_array)
                
                if len(time_indices) > 0:
                    # Extract all data points within the time range
                    start_idx = time_indices[0]
                    end_idx = time_indices[-1]
                    
                    segments[state] = {
                        'data': data[time_indices],  # All matching indices
                        'time': time_array[time_indices],  # Corresponding time points
                        'start_idx': int(start_idx),
                        'end_idx': int(end_idx),
                        'duration': movement_end - movement_start,  # Original duration
                        'data_type': data_type,
                        'n_points': len(time_indices),  # Actual number of data points
                        'movement_time': [movement_start, movement_end],  # Original movement time
                        'aligned_time': [aligned_start, aligned_end],  # Aligned time
                        'data_start_time': data_start_time  # This data's start time offset
                    }
                    print(f"  {state}: {data_type} movement[{movement_start:.2f}:{movement_end:.2f}s] "
                          f"+ offset({data_start_time:.2f}s) "
                          f"-> aligned[{aligned_start:.2f}:{aligned_end:.2f}s] "
                          f"-> indices[{start_idx}:{end_idx}] "
                          f"({len(time_indices)} points)")
                else:
                    print(f"  {state}: {data_type} no data points found in aligned time range "
                          f"[{aligned_start:.2f}:{aligned_end:.2f}s] "
                          f"(movement: [{movement_start:.2f}:{movement_end:.2f}s] + offset: {data_start_time:.2f}s)")
                    segments[state] = None
            else:
                segments[state] = None
        
        return segments
    
    def segment_dataset(self, dataset_path: str) -> Dict:
        """Segment all data for one dataset with time alignment"""
        dataset_name = os.path.basename(dataset_path)
        print(f"Segmenting dataset: {dataset_name}")
        
        # Load movement segmentation results
        movement_results = self.load_movement_results(dataset_path)
        if movement_results is None:
            return {'error': 'No movement results found'}
        
        # Load EMG timestamps for time alignment
        emg_timestamps = self.load_emg_timestamps(dataset_path)
        if emg_timestamps is None:
            print("Warning: No EMG timestamps found - EMG data alignment may be incorrect")
        
        result = {
            'dataset_name': dataset_name,
            'movement_states': movement_results,
            'eeg_segments': {},
            'emg_segments': {},
            'available_states': []
        }
        
        # Determine available movement states
        for state in self.states:
            if state in movement_results and movement_results[state] is not None:
                result['available_states'].append(state)
        
        print(f"Available states: {result['available_states']}")
        
        # Segment EEG data with time alignment
        eeg_data = self.load_eeg_data(dataset_path)
        for eeg_name, eeg_info in eeg_data.items():
            print(f"Segmenting EEG: {eeg_name}")
            eeg_segments = self.segment_data_by_time(
                eeg_info['data'], eeg_info['time'], movement_results, f"EEG-{eeg_name}"
            )
            
            # Add channel information
            for state, segment in eeg_segments.items():
                if segment is not None:
                    segment['channels'] = eeg_info['channels']
            
            result['eeg_segments'][eeg_name] = eeg_segments
        
        # Segment EMG data (firing patterns and CWF) with time alignment
        if emg_timestamps is not None:
            npz_data = self.load_npz_data(dataset_path)
            
            for mode, mode_data in npz_data.items():
                print(f"Segmenting EMG mode: {mode}")
                mode_segments = {}
                
                # Process each channel's data
                for ch_label, ch_data in mode_data.items():
                    if not isinstance(ch_data, dict):
                        continue  # Skip non-channel data
                    
                    ch_segments = {}
                    
                    # Segment firing patterns with time alignment
                    if 'firing_pattern' in ch_data:
                        fp_data = ch_data['firing_pattern']
                        if isinstance(fp_data, np.ndarray) and len(fp_data) == len(emg_timestamps):
                            segments = self.segment_data_by_time(
                                fp_data, emg_timestamps, movement_results, 
                                f"FiringPattern-{ch_label}"
                            )
                            ch_segments['firing_pattern'] = segments
                    
                    # Segment CWF with time alignment
                    if 'CWF' in ch_data:
                        cwf_data = ch_data['CWF']
                        if isinstance(cwf_data, np.ndarray) and len(cwf_data) == len(emg_timestamps):
                            segments = self.segment_data_by_time(
                                cwf_data, emg_timestamps, movement_results, 
                                f"CWF-{ch_label}"
                            )
                            ch_segments['CWF'] = segments
                    
                    # Note: CWF_comp and recon_total are not segmented per user requirements
                    # Only firing_pattern and CWF require time alignment
                    
                    if ch_segments:
                        mode_segments[ch_label] = ch_segments
                
                if mode_segments:
                    result['emg_segments'][mode] = mode_segments
                    print(f"  {mode}: segmented {len(mode_segments)} channels")
        else:
            print("No EMG timestamps found - skipping EMG segmentation")
        
        return result
    
    def plot_firing_pattern_and_cwf_combined(self, segments: Dict, output_folder: str = None):
        """Create combined plots with firing patterns (by column) and CWF as subplots"""
        dataset_name = segments['dataset_name']
        available_states = segments['available_states']
        
        if not available_states:
            print("No states available for firing pattern visualization")
            return
        
        colors = {'sit_to_stand': 'blue', 'stand_to_walk': 'green', 
                 'steady_gait': 'cyan', 'walk_to_stop': 'magenta'}
        
        # Plot firing patterns and CWF for each channel
        for mode, mode_segments in segments['emg_segments'].items():
            for ch_label, ch_segments in mode_segments.items():
                
                # Determine what data we have
                has_fp = 'firing_pattern' in ch_segments
                has_cwf = 'CWF' in ch_segments
                
                if not has_fp and not has_cwf:
                    continue
                
                # Determine number of firing pattern components
                max_components = 0
                if has_fp:
                    fp_segments = ch_segments['firing_pattern']
                    for state in available_states:
                        if state in fp_segments and fp_segments[state] is not None:
                            segment = fp_segments[state]
                            data = segment['data']
                            if data.ndim > 1:
                                max_components = max(max_components, data.shape[1])
                            else:
                                max_components = max(max_components, 1)
                
                # Calculate total subplots needed
                total_subplots = max_components + (1 if has_cwf else 0)
                
                if total_subplots == 0:
                    continue
                
                # Create subplots
                fig, axes = plt.subplots(total_subplots, 1, figsize=(15, 4 * total_subplots))
                if total_subplots == 1:
                    axes = [axes]
                
                subplot_idx = 0
                
                # Plot each firing pattern component
                if has_fp:
                    for fp_idx in range(max_components):
                        ax = axes[subplot_idx]
                        
                        has_data = False
                        for state in available_states:
                            if state in fp_segments and fp_segments[state] is not None:
                                segment = fp_segments[state]
                                time = segment['time']
                                data = segment['data']
                                
                                # Extract specific firing pattern component
                                if data.ndim > 1 and fp_idx < data.shape[1]:
                                    y_data = data[:, fp_idx]
                                elif data.ndim == 1 and fp_idx == 0:
                                    y_data = data
                                else:
                                    continue  # Skip if component doesn't exist
                                
                                ax.plot(time, y_data, color=colors[state], 
                                       label=f'{state}', alpha=0.8, linewidth=1.5)
                                has_data = True
                        
                        if has_data:
                            ax.set_ylabel('Coefficient (a.u. per sample)')
                            ax.set_title(f'Firing Pattern {fp_idx+1}')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                        
                        subplot_idx += 1
                
                # Plot CWF
                if has_cwf:
                    ax = axes[subplot_idx]
                    cwf_segments = ch_segments['CWF']
                    
                    has_data = False
                    for state in available_states:
                        if state in cwf_segments and cwf_segments[state] is not None:
                            segment = cwf_segments[state]
                            time = segment['time']
                            data = segment['data']
                            
                            # Handle multidimensional data
                            if data.ndim > 1:
                                y_data = np.mean(data, axis=1)
                            else:
                                y_data = data
                            
                            ax.plot(time, y_data, color=colors[state], 
                                   label=f'{state}', alpha=0.8, linewidth=1.5)
                            has_data = True
                    
                    if has_data:
                        ax.set_ylabel('CWF (µV)')
                        ax.set_title('Total CWF')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                
                # Set common x-label for bottom subplot
                axes[-1].set_xlabel('Time (s)')
                
                # Overall title
                plt.suptitle(f'{ch_label} - Firing Patterns & CWF ({mode} mode) - {dataset_name}', 
                            fontsize=14, y=0.98)
                plt.tight_layout()
                
                if output_folder:
                    os.makedirs(output_folder, exist_ok=True)
                    save_path = os.path.join(output_folder, 
                                           f'Combined_{ch_label}_{mode}_{dataset_name}.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Combined plot saved: {save_path}")
                
                plt.show()
    
    def visualize_segments(self, segments: Dict, output_folder: str = None):
        """Create visualization of segmented data with time alignment"""
        dataset_name = segments['dataset_name']
        available_states = segments['available_states']
        
        if not available_states:
            print("No states available for visualization")
            return
        
        colors = {'sit_to_stand': 'blue', 'stand_to_walk': 'green', 
                 'steady_gait': 'cyan', 'walk_to_stop': 'magenta'}
        
        # Plot EEG segments with time alignment
        for eeg_name, eeg_segments in segments['eeg_segments'].items():
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            
            for state in available_states:
                if state in eeg_segments and eeg_segments[state] is not None:
                    segment = eeg_segments[state]
                    time = segment['time']
                    data = segment['data']
                    
                    # Plot average of all channels
                    if data.ndim > 1:
                        y_data = np.mean(data, axis=1)
                    else:
                        y_data = data
                    
                    ax.plot(time, y_data, color=colors[state], 
                           label=f'{state}', alpha=0.8, linewidth=1.5)
            
            ax.set_ylabel(f'Zscore Amplitude (µV)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'EEG Segments: {eeg_name} - {dataset_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                save_path = os.path.join(output_folder, f'EEG_{eeg_name}_segments_{dataset_name}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"EEG plot saved: {save_path}")
            
            plt.tight_layout()
            plt.show()
        
        # Create combined firing pattern and CWF plots
        self.plot_firing_pattern_and_cwf_combined(segments, output_folder)
    
    def save_segments(self, segments: Dict, output_folder: str):
        """Save segmented data with time alignment information"""
        os.makedirs(output_folder, exist_ok=True)
        dataset_name = segments['dataset_name']
        
        # Save complete segments with time alignment info
        segments_path = os.path.join(output_folder, f'segments_{dataset_name}.pkl')
        with open(segments_path, 'wb') as f:
            pickle.dump(segments, f)
        print(f"Segments saved: {segments_path}")
        
        # Save detailed summary CSV including time alignment information
        summary_data = []
        
        # EEG summary with time alignment
        for eeg_name, eeg_segments in segments['eeg_segments'].items():
            for state in self.states:
                row = {
                    'dataset': dataset_name,
                    'data_type': 'EEG',
                    'channel/file': eeg_name,
                    'state': state,
                    'available': state in segments['available_states'],
                    'has_data': state in eeg_segments and eeg_segments[state] is not None
                }
                if row['has_data']:
                    segment = eeg_segments[state]
                    row.update({
                        'n_samples': len(segment['data']),
                        'duration_s': segment['duration'],
                        'n_channels': len(segment.get('channels', [])),
                        'start_time_s': segment['time'][0] if len(segment['time']) > 0 else None,
                        'end_time_s': segment['time'][-1] if len(segment['time']) > 0 else None,
                        'start_idx': segment['start_idx'],
                        'end_idx': segment['end_idx'],
                        'n_points': segment['n_points'],
                        'movement_start': segment['movement_time'][0],
                        'movement_end': segment['movement_time'][1],
                        'aligned_start': segment['aligned_time'][0],
                        'aligned_end': segment['aligned_time'][1],
                        'data_start_offset': segment['data_start_time']
                    })
                else:
                    row.update({
                        'n_samples': None, 'duration_s': None, 'n_channels': None,
                        'start_time_s': None, 'end_time_s': None, 'start_idx': None,
                        'end_idx': None, 'n_points': None, 'movement_start': None,
                        'movement_end': None, 'aligned_start': None, 'aligned_end': None,
                        'data_start_offset': None
                    })
                summary_data.append(row)
        
        # EMG summary with time alignment
        for mode, mode_segments in segments['emg_segments'].items():
            for ch_label, ch_segments in mode_segments.items():
                for data_type in ['firing_pattern', 'CWF']:  # Only these two types
                    if data_type not in ch_segments:
                        continue
                    
                    data_segments = ch_segments[data_type]
                    for state in self.states:
                        row = {
                            'dataset': dataset_name,
                            'data_type': f'EMG_{mode}_{data_type}',
                            'channel/file': ch_label,
                            'state': state,
                            'available': state in segments['available_states'],
                            'has_data': state in data_segments and data_segments[state] is not None
                        }
                        if row['has_data']:
                            segment = data_segments[state]
                            row.update({
                                'n_samples': len(segment['data']),
                                'duration_s': segment['duration'],
                                'n_channels': segment['data'].shape[1] if segment['data'].ndim > 1 else 1,
                                'start_time_s': segment['time'][0] if len(segment['time']) > 0 else None,
                                'end_time_s': segment['time'][-1] if len(segment['time']) > 0 else None,
                                'start_idx': segment['start_idx'],
                                'end_idx': segment['end_idx'],
                                'n_points': segment['n_points'],
                                'movement_start': segment['movement_time'][0],
                                'movement_end': segment['movement_time'][1],
                                'aligned_start': segment['aligned_time'][0],
                                'aligned_end': segment['aligned_time'][1],
                                'data_start_offset': segment['data_start_time']
                            })
                        else:
                            row.update({
                                'n_samples': None, 'duration_s': None, 'n_channels': None,
                                'start_time_s': None, 'end_time_s': None, 'start_idx': None,
                                'end_idx': None, 'n_points': None, 'movement_start': None,
                                'movement_end': None, 'aligned_start': None, 'aligned_end': None,
                                'data_start_offset': None
                            })
                        summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_folder, f'segmentation_summary_{dataset_name}.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"Segmentation summary saved: {csv_path}")
    
    def process_all_datasets(self, base_path: str, prefix: str = 'path2_') -> Dict:
        """Process all datasets and save results in individual dataset folders"""
        datasets = sorted([d for d in glob.glob(os.path.join(base_path, f'{prefix}*')) 
                          if os.path.isdir(d)])
        
        if not datasets:
            print(f"No {prefix} datasets found in {base_path}")
            return {}
        
        all_results = {}
        
        for i, dataset_path in enumerate(datasets, 1):
            dataset_name = os.path.basename(dataset_path)
            print(f"\n--- Processing {i}/{len(datasets)}: {dataset_name} ---")
            
            try:
                segments = self.segment_dataset(dataset_path)
                
                if 'error' not in segments:
                    all_results[dataset_name] = segments
                    
                    # Create output folder WITHIN the dataset folder
                    dataset_output = os.path.join(dataset_path, 'neural_movement_segmentation')
                    
                    # Save and visualize with time alignment
                    self.save_segments(segments, dataset_output)
                    self.visualize_segments(segments, dataset_output)
                    
                    print(f"✓ Successfully processed {dataset_name}")
                    print(f"  Results saved in: {dataset_output}")
                else:
                    print(f"✗ Failed to process {dataset_name}: {segments['error']}")
            
            except Exception as e:
                print(f"✗ Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save master summary in base path
        if all_results:
            master_summary = []
            for name, result in all_results.items():
                eeg_count = len(result['eeg_segments'])
                emg_modes = len(result['emg_segments'])
                emg_channels = sum(len(mode_segs) for mode_segs in result['emg_segments'].values())
                
                row = {
                    'dataset': name,
                    'available_states': len(result['available_states']),
                    'states_list': ', '.join(result['available_states']),
                    'eeg_files': eeg_count,
                    'emg_modes': emg_modes,
                    'emg_channels': emg_channels
                }
                master_summary.append(row)
            
            master_df = pd.DataFrame(master_summary)
            master_csv = os.path.join(base_path, 'movement_segmentation_master_summary.csv')
            master_df.to_csv(master_csv, index=False)
            print(f"\nMaster summary saved: {master_csv}")
        
        return all_results


# Usage functions
def segment_single_dataset(dataset_path: str, output_folder: str = None):
    """Segment single dataset with time alignment"""
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
    """Segment all datasets with time alignment"""
    segmentor = MovementDataSegmentor()
    return segmentor.process_all_datasets(base_path)


# Example usage
if __name__ == "__main__":
    # Set your data path
    BASE_PATH = '/home/ntk/exo/ExoskeletonPowerAsistance/simulation/mocap_EMG_EEG_data/20241225'
    
    print("=== Movement Data Segmentation with Time Alignment ===")
    
    # Process all datasets (saves results in individual dataset folders)
    all_results = segment_all_datasets(BASE_PATH)
    
    if all_results:
        print(f"\n✓ Successfully processed {len(all_results)} datasets")
        
        # Print summary
        for name, result in all_results.items():
            states = result['available_states']
            eeg_count = len(result['eeg_segments'])
            emg_modes = list(result['emg_segments'].keys())
            print(f"  {name}: {len(states)} states, {eeg_count} EEG files, EMG modes: {emg_modes}")
    
    # Example: Process single dataset
    # single_path = os.path.join(BASE_PATH, 'path2_01')
    # result = segment_single_dataset(single_path)

# %%
