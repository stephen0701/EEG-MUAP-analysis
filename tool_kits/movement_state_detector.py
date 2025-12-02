# %%
"""
Integrated Movement State Segmentation Based on Segmental Weighted Center of Mass
Combines original movement segmentation with segmental COM calculation method
Detects four states: sit-to-stand, stand-to-walk, steady-gait, walk-to-stop
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-friendly
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, convolve, decimate, butter, filtfilt
from scipy.interpolate import PchipInterpolator
import os
from typing import Dict, List, Tuple, Optional
import glob
import warnings
import pickle

class IntegratedMovementStateSegmentor:
    """
    Integrated Movement State Segmentor
    Uses segmental weighted method to calculate center of mass
    Detects four states: sit-to-stand, stand-to-walk, steady-gait, walk-to-stop
    """
    
    def __init__(self, fs: int = 30):
        self.fs = fs  # sampling frequency (frames per second)
        self.movement_states = {
            'sit_to_stand': [],    # sit-to-stand (T1)
            'stand_to_walk': [],   # stand-to-walk (T2) 
            'steady_gait': [],     # steady gait (T4)
            'walk_to_stop': []     # walk-to-stop (T3)
        }
        
        # Default human body segment parameters (based on anthropometric data)
        self.default_segments = {
            "Upper_arm_R": ("RShoulder", "RElbow", 0.436, 0.028),
            "Upper_arm_L": ("LShoulder", "LElbow", 0.436, 0.028),
            "Forearm_R": ("RElbow", "RWrist", 0.682, 0.022),
            "Forearm_L": ("LElbow", "LWrist", 0.682, 0.022),
            "Foot_R": ("RAnkle", ("RBigToe", "RSmallToe"), 0.5, 0.0145),
            "Foot_L": ("LAnkle", ("LBigToe", "LSmallToe"), 0.5, 0.0145),
            "Leg_R": ("RKnee", "RAnkle", 0.433, 0.0465),
            "Leg_L": ("LKnee", "LAnkle", 0.433, 0.0465),
            "Thigh_R": ("RHip", "RKnee", 0.433, 0.1),
            "Thigh_L": ("LHip", "LKnee", 0.433, 0.1),
            "Head": ("Neck", "Nose", 1, 0.081),
            "Trunk_R": ("RHip", "RShoulder", 0.5, 0.2485),
            "Trunk_L": ("LHip", "LShoulder", 0.5, 0.2485)
        }
    
    def load_and_clean_trc_data(self, trc_path: str) -> Tuple[pd.DataFrame, dict]:
        """
        Enhanced TRC file loading function
        Returns: (trc_data DataFrame, camera_info dict)
        """
        try:
            # Read first 4 header lines
            with open(trc_path, 'r') as f:
                header_lines = [next(f) for _ in range(4)]
            
            # Parse camera info (lines 2-3)
            camera_info_header = header_lines[1].strip().split('\t')
            camera_info_values = header_lines[2].strip().split('\t')
            if len(camera_info_header) == len(camera_info_values):
                camera_info = dict(zip(camera_info_header, camera_info_values))
            else:
                camera_info = {"raw_header_lines": header_lines}
            
            # Marker names usually in line 4
            header_line = header_lines[3].strip().split('\t')
            cleaned_header = [col for col in header_line if col != '']
            
            # Build column names: Frame#, Time, then markers * 3 axes
            column_names = ['Frame#', 'Time'] + [f'{marker}_{axis}' 
                           for marker in cleaned_header[2:] for axis in ['X','Y','Z']]
            
            # Read numerical data (skip first 5 lines)
            trc_data = pd.read_csv(trc_path, sep='\t', skiprows=5, header=None)
            trc_data.columns = column_names[:len(trc_data.columns)]
            
            # Convert to numeric and remove empty columns
            trc_data = trc_data.apply(pd.to_numeric, errors='coerce')
            trc_data = trc_data.dropna(axis=1, how='all')
            
            return trc_data, camera_info
            
        except Exception as e:
            print(f"Error loading TRC file {trc_path}: {e}")
            return None, None
    
    def cal_seg_COM_trajectory(self, start_coords: np.ndarray, end_coords: np.ndarray, 
                              position_ratio: float, mass_ratio: float) -> np.ndarray:
        """
        Calculate single segment's center of mass trajectory contribution
        start_coords, end_coords: (N x 3) coordinate arrays
        position_ratio: position ratio from proximal to distal (0..1)
        mass_ratio: mass ratio
        returns: (N x 3) this segment's contribution to whole-body COM (mass-weighted)
        """
        segment_COM = (start_coords + (end_coords - start_coords) * position_ratio) * mass_ratio
        return segment_COM
    
    def calculate_all_COMs(self, trc_data: pd.DataFrame, segments: dict = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate whole-body center of mass using segmental weighted method
        trc_data: DataFrame containing marker columns NAME_X/Y/Z
        segments: segment definition dictionary (segment_name -> (start, end_or_tuple, position_ratio, mass_ratio))
        returns: dictionary containing each segment's COM and 'Total_COM'
        """
        if segments is None:
            segments = self.default_segments
        
        num_timesteps = trc_data.shape[0]
        total_COM = np.zeros((num_timesteps, 3))
        segment_COMs = {}
        
        for segment, (start, end, position_ratio, mass_ratio) in segments.items():
            try:
                # Get end coordinates
                if isinstance(end, tuple):
                    # Average two markers as distal endpoint (e.g., toes)
                    coords0 = trc_data[[f"{end[0]}_X", f"{end[0]}_Y", f"{end[0]}_Z"]].values
                    coords1 = trc_data[[f"{end[1]}_X", f"{end[1]}_Y", f"{end[1]}_Z"]].values
                    end_coords = (coords0 + coords1) / 2.0
                else:
                    end_coords = trc_data[[f"{end}_X", f"{end}_Y", f"{end}_Z"]].values
                
                start_coords = trc_data[[f"{start}_X", f"{start}_Y", f"{start}_Z"]].values
                
                # Defensive check
                if start_coords.shape[0] != num_timesteps or end_coords.shape[0] != num_timesteps:
                    warnings.warn(f"Marker array length mismatch for segment {segment}")
                    continue
                
                # Check for NaN values
                if np.any(np.isnan(start_coords)) or np.any(np.isnan(end_coords)):
                    warnings.warn(f"NaN values found in segment {segment}, skipping")
                    continue
                
                seg_contrib = self.cal_seg_COM_trajectory(start_coords, end_coords, 
                                                        position_ratio, mass_ratio)
                segment_COMs[segment] = pd.DataFrame(seg_contrib, columns=['X','Y','Z'])
                total_COM += seg_contrib
                
            except KeyError as e:
                warnings.warn(f"Missing marker for segment {segment}: {e}")
                continue
            except Exception as e:
                warnings.warn(f"Error calculating segment {segment}: {e}")
                continue
        
        # Store total COM as DataFrame
        segment_COMs['Total_COM'] = pd.DataFrame(total_COM, columns=['X','Y','Z'])
        return segment_COMs
    
    def compute_vcm_segmental(self, trc_data: pd.DataFrame, SR: int = 30, SRd: int = 30, 
                             lowpass_cut: float = 6.0, filter_order: int = 4, 
                             segments: dict = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Compute whole-body center of mass velocity using segmental method
        Returns: (v, t, cm_df) - COM velocity, time vector, filtered COM position
        """
        # Calculate segmental COMs
        segs = self.calculate_all_COMs(trc_data, segments=segments)
        cm_raw = segs['Total_COM'].values  # Nx3
        
        # Time vectors
        N = cm_raw.shape[0]
        t_old = np.arange(N) / SR
        
        # Resample/downsample
        if abs(SRd - SR) < 1e-6:
            cm = cm_raw.copy()
            Fs_actual = SR
            t = t_old.copy()
        elif SRd < SR:
            down_factor = int(round(SR / SRd))
            if down_factor < 1:
                down_factor = 1
            cm = np.column_stack([decimate(cm_raw[:,i], down_factor, ftype='iir', zero_phase=True) 
                                for i in range(3)])
            Fs_actual = SR / down_factor
            t = np.arange(cm.shape[0]) / Fs_actual
        else:
            # Upsample using pchip interpolation
            Nnew = int(round((N - 1) * SRd / SR)) + 1
            t = np.linspace(0, (N - 1) / SR, Nnew)
            cm = np.vstack([PchipInterpolator(t_old, cm_raw[:,i])(t) for i in range(3)]).T
            Fs_actual = SRd
        
        # Lowpass filtering for smoothing
        nyq = Fs_actual / 2.0
        if lowpass_cut >= nyq or lowpass_cut <= 0:
            cm_f = cm
        else:
            b, a = butter(filter_order, lowpass_cut / nyq)
            cm_f = np.column_stack([filtfilt(b, a, cm[:,i]) for i in range(3)])
        
        # Calculate velocity: use horizontal plane (X-Z)
        x = cm_f[:,0]
        z = cm_f[:,2]
        vx = np.gradient(x) * Fs_actual
        vz = np.gradient(z) * Fs_actual
        v = np.sqrt(vx**2 + vz**2)
        
        # Short-term smoothing to reduce spikes (50ms window)
        win_samples = max(1, int(round(0.05 * Fs_actual)))
        if win_samples > 1:
            v = pd.Series(v).rolling(window=win_samples, center=True, min_periods=1).mean().to_numpy()
        
        cm_df = pd.DataFrame(cm_f, columns=['X','Y','Z'])
        return v, t, cm_df
    
    def analyze_transient_segments(self, vcm_data: np.ndarray, time_vector: np.ndarray, 
                                 data_label: str = 'unknown', 
                                 output_folder: str = None) -> Dict:
        """
        Analyze transient segments: sit→stand, stand→walk, steady gait, walk→stop
        Integrates original algorithm with new segmental COM calculation
        """
        transient_results = {}
        vcm_data = vcm_data.flatten()
        time_vector = time_vector.flatten()
        
        if len(vcm_data) == 0:
            warnings.warn("vcm_data is empty")
            return {'T1': None, 'T2': None, 'T3': None, 'steady': None}
        
        # Ensure time vector and data lengths match
        if len(time_vector) != len(vcm_data):
            warnings.warn("Time vector and VCM data length mismatch")
            time_vector = np.arange(len(vcm_data)) / self.fs
        
        # Create single figure with valley and peak points
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot original VCM data
        ax.plot(time_vector, vcm_data, 'k-', linewidth=1.5, label='VCM (Segmental)')
        
        # h1 filter (approximate derivative emphasis)
        h1 = np.array([-1, -2, 0, 2, 1], dtype=float)
        y = convolve(vcm_data, h1, mode='same')
        
        # Calculate energy and normalize
        y2 = y**2
        y2n = y2 if np.max(y2) == 0 else (y2 / np.max(y2))
        
        # Moving average smoothing (~1 second window)
        window_size = min(31, len(y2n))
        if window_size > 0:
            y3 = convolve(y2n, np.ones(window_size) / window_size, mode='same')
            if np.max(y3) != 0:
                y3 = y3 / np.max(y3)
        else:
            y3 = y2n
        
        # Detect active regions
        active = y3 > np.mean(y3)
        padded = np.concatenate([[False], active, [False]]).astype(int)
        diff_set = np.diff(padded)
        left_indices = np.where(diff_set == 1)[0]
        right_indices = np.where(diff_set == -1)[0] - 1
        
        # Find valley points
        valley_locations = []
        for li in left_indices:
            start_search = max(0, li - self.fs)
            segment = vcm_data[start_search:li+1]
            if segment.size > 0:
                min_pos = np.argmin(segment)
                valley_locations.append(start_search + int(min_pos))
        
        # Add last valley
        if len(right_indices) > 0:
            start_search = max(0, right_indices[-1] - self.fs)
            segment = vcm_data[start_search:right_indices[-1]+1]
            if segment.size > 0:
                min_pos = np.argmin(segment)
                valley_locations.append(start_search + int(min_pos))
        
        valley_locations = sorted(list(set(valley_locations)))  # Remove duplicates and sort
        
        # Mark valley points on plot
        if len(valley_locations) > 0:
            ax.plot(time_vector[valley_locations], vcm_data[valley_locations], 
                   'ro', markersize=8, markerfacecolor='red', label='Valley Points')
        
        # If enough valley points found, perform state segmentation
        legend_labels = []
        if len(valley_locations) >= 3:
            print(f"Found {len(valley_locations)} valley points for segmentation")
            
            X1 = valley_locations[1]  # Second valley
            X2 = valley_locations[2]  # Third valley
            
            if X2 > X1:
                # Find peaks between second and third valleys
                sub_vcm = vcm_data[X1:X2+1]
                if len(sub_vcm) > 0:
                    # Dynamic threshold setting
                    #threshold = np.min(sub_vcm) + 0.7 * (np.max(sub_vcm) - np.min(sub_vcm))
                    #threshold = np.percentile(sub_vcm, 75)
                    threshold = np.mean(sub_vcm) + 0.3 * np.std(sub_vcm)
                    min_distance = max(1, int(0.3 * self.fs))
                    
                    peaks, _ = find_peaks(sub_vcm, height=threshold, distance=min_distance)
                    
                    if len(peaks) >= 2:
                        # Identify first and last peak indices
                        X1_peak = X1 + peaks[0]
                        X2_peak = X1 + peaks[-1]
                                         
                        # Mark peak points on plot
                        ax.plot(time_vector[X1_peak], vcm_data[X1_peak], 
                               'go', markersize=8, markerfacecolor='green', label='Peak Points')
                        ax.plot(time_vector[X2_peak], vcm_data[X2_peak], 
                               'go', markersize=8, markerfacecolor='green')
                        
                        # Define time intervals for four states
                        T1_start, T1_end = valley_locations[0], valley_locations[1]    # sit→stand
                        T2_start, T2_end = valley_locations[1], X1_peak               # stand→walk
                        T4_start, T4_end = X1_peak, X2_peak                          # steady gait
                        T3_start, T3_end = X2_peak, valley_locations[-1]             # walk→stop
                        
                        # Ensure indices are within valid range
                        def clamp_index(idx):
                            return max(0, min(idx, len(time_vector)-1))
                        
                        T1_start, T1_end = clamp_index(T1_start), clamp_index(T1_end)
                        T2_start, T2_end = clamp_index(T2_start), clamp_index(T2_end)
                        T3_start, T3_end = clamp_index(T3_start), clamp_index(T3_end)
                        T4_start, T4_end = clamp_index(T4_start), clamp_index(T4_end)
                        
                        # Mark state intervals on plot with colored background regions
                        colors = {'sit_to_stand': 'blue', 'stand_to_walk': 'green', 
                                 'steady_gait': 'cyan', 'walk_to_stop': 'magenta'}
                        
                        # Add colored background regions and create legend labels
                        ax.axvspan(time_vector[T1_start], time_vector[T1_end], 
                                  alpha=0.2, color=colors['sit_to_stand'])
                        legend_labels.append(f'Sit→Stand ({time_vector[T1_start]:.1f}-{time_vector[T1_end]:.1f}s)')
                        
                        ax.axvspan(time_vector[T2_start], time_vector[T2_end], 
                                  alpha=0.2, color=colors['stand_to_walk'])
                        legend_labels.append(f'Stand→Walk ({time_vector[T2_start]:.1f}-{time_vector[T2_end]:.1f}s)')
                        
                        ax.axvspan(time_vector[T4_start], time_vector[T4_end], 
                                  alpha=0.2, color=colors['steady_gait'])
                        legend_labels.append(f'Steady Gait ({time_vector[T4_start]:.1f}-{time_vector[T4_end]:.1f}s)')
                        
                        ax.axvspan(time_vector[T3_start], time_vector[T3_end], 
                                  alpha=0.2, color=colors['walk_to_stop'])
                        legend_labels.append(f'Walk→Stop ({time_vector[T3_start]:.1f}-{time_vector[T3_end]:.1f}s)')
                        
                        # Output results
                        print(f'Sit→Stand (T1): {time_vector[T1_start]:.2f} to {time_vector[T1_end]:.2f} seconds')
                        print(f'Stand→Walk (T2): {time_vector[T2_start]:.2f} to {time_vector[T2_end]:.2f} seconds') 
                        print(f'Steady Gait (T4): {time_vector[T4_start]:.2f} to {time_vector[T4_end]:.2f} seconds')
                        print(f'Walk→Stop (T3): {time_vector[T3_start]:.2f} to {time_vector[T3_end]:.2f} seconds')
                        
                        # Store results
                        transient_results = {
                            'T1': (float(time_vector[T1_start]), float(time_vector[T1_end])),
                            'T2': (float(time_vector[T2_start]), float(time_vector[T2_end])),
                            'T3': (float(time_vector[T3_start]), float(time_vector[T3_end])),
                            'steady': (float(time_vector[T4_start]), float(time_vector[T4_end])),
                            'sit_to_stand': {
                                'start_time': float(time_vector[T1_start]),
                                'end_time': float(time_vector[T1_end]),
                                'duration': float(time_vector[T1_end] - time_vector[T1_start]),
                                'start_idx': int(T1_start),
                                'end_idx': int(T1_end)
                            },
                            'stand_to_walk': {
                                'start_time': float(time_vector[T2_start]),
                                'end_time': float(time_vector[T2_end]),
                                'duration': float(time_vector[T2_end] - time_vector[T2_start]),
                                'start_idx': int(T2_start),
                                'end_idx': int(T2_end)
                            },
                            'steady_gait': {
                                'start_time': float(time_vector[T4_start]),
                                'end_time': float(time_vector[T4_end]),
                                'duration': float(time_vector[T4_end] - time_vector[T4_start]),
                                'start_idx': int(T4_start),
                                'end_idx': int(T4_end),
                                'n_peaks': len(peaks),
                                'mean_velocity': float(np.mean(vcm_data[T4_start:T4_end+1])),
                                'velocity_std': float(np.std(vcm_data[T4_start:T4_end+1]))
                            },
                            'walk_to_stop': {
                                'start_time': float(time_vector[T3_start]),
                                'end_time': float(time_vector[T3_end]),
                                'duration': float(time_vector[T3_end] - time_vector[T3_start]),
                                'start_idx': int(T3_start),
                                'end_idx': int(T3_end)
                            },
                            'metadata': {
                                'data_label': data_label,
                                'total_duration': float(time_vector[-1] - time_vector[0]),
                                'sampling_rate': self.fs,
                                'n_valley_points': len(valley_locations),
                                'n_gait_peaks': len(peaks),
                                'method': 'segmental_weighted_COM'
                            }
                        }
                        
                    else:
                        warnings.warn("Not enough peaks found in steady state region")
                        transient_results = {'T1': None, 'T2': None, 'T3': None, 'steady': None}
                else:
                    warnings.warn("No data points between X1 and X2")
                    transient_results = {'T1': None, 'T2': None, 'T3': None, 'steady': None}
            else:
                warnings.warn("Valley indices invalid (X2 <= X1)")
                transient_results = {'T1': None, 'T2': None, 'T3': None, 'steady': None}
        else:
            warnings.warn(f"Only found {len(valley_locations)} valley points, cannot perform complete state segmentation")
            transient_results = {'T1': None, 'T2': None, 'T3': None, 'steady': None}
        
        # Finalize plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('VCM (m/s)')
        ax.set_title(f'Segmental Center of Mass Velocity Analysis ({data_label})')
        ax.grid(True, alpha=0.3)
        
        # Create comprehensive legend with colored patches
        handles, labels = ax.get_legend_handles_labels()
        
        # Add colored patches for each state interval
        if len(valley_locations) >= 3 and len(legend_labels) > 0:
            state_colors = ['blue', 'green', 'cyan', 'magenta']  # Match the colors used in axvspan
            for i, label in enumerate(legend_labels):
                color = state_colors[i] if i < len(state_colors) else 'gray'
                handles.append(plt.Rectangle((0,0),1,1, fc=color, alpha=0.3, ec='none'))
                labels.append(label)
        
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save figure directly to output folder (no subfolder)
        if output_folder and os.path.exists(output_folder):
            save_path = os.path.join(output_folder, f'segmental_transitional_stages_{data_label}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return transient_results
    
    def load_sync_time_markers(self, base_path: str, prefix: str = 'path2_') -> Dict:
        """
        Load synchronization time markers
        """
        folders = [d for d in os.listdir(base_path) 
                  if d.startswith(prefix) and os.path.isdir(os.path.join(base_path, d))]
        sync_dict = {}
        
        for name in folders:
            npz_path = os.path.join(base_path, name, 'opensim', 'sync_time_marker.npz')
            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path, allow_pickle=True) as npz:
                        items = []
                        for k in npz.files:
                            items.append(npz[k])
                        # Fix: should modify items[0] (sync_timeline), not items[1] (marker)
                        if len(items) >= 2 and isinstance(items[0], np.ndarray) and items[0].size > 0:
                            arr = items[0].astype(float).copy().reshape(-1)
                            new_val = arr[-1] + 0.0333
                            items[0] = np.concatenate([arr, np.array([new_val])])
                        sync_dict[name] = items
                except Exception as e:
                    warnings.warn(f"Failed to load {npz_path}: {e}")
                    sync_dict[name] = None
            else:
                sync_dict[name] = None
        return sync_dict
    '''
    def count_videotime_from_sync(self, sync_entry) -> Optional[np.ndarray]:
        if sync_entry is None or len(sync_entry) < 2:
            return None
        try:
            frame_time_data = np.array(sync_entry[1]).astype(float).reshape(-1)
            start_time_arr = np.array(sync_entry[0]).astype(float).reshape(-1)
            start_time = start_time_arr[0]
            timeOfvideo = (frame_time_data - start_time).T  # Row vector like MATLAB
            return timeOfvideo
        except Exception:
            return None
    '''
        
    def count_videotime_from_sync(self, sync_entry) -> Optional[np.ndarray]:
        """
        Calculate video time from sync data
        sync_entry[0] = sync_timeline (frame timestamps array)
        sync_entry[1] = marker (start/end markers only)
        """
        if sync_entry is None or len(sync_entry) < 1:
            return None
        try:
            # Use sync_timeline directly (sync_entry[0])
            # Note: load_sync_time_markers already adds one extra frame (last + 0.0333s)
            frame_time_data = np.array(sync_entry[0]).astype(float).reshape(-1)
            timeOfvideo = frame_time_data.T  # Use directly as time vector
            return timeOfvideo
        except Exception:
            return None
    
    def process_single_file(self, trc_path: str, folder_name: str, sync_time_data: np.ndarray = None, 
                          data_label: str = None, base_output_folder: str = None) -> Dict:
        """
        Process single TRC file (using segmental weighted method)
        """
        if data_label is None:
            data_label = folder_name
        
        print(f"Processing with segmental method: {data_label}")
        
        # Load TRC data
        trc_data, camera_info = self.load_and_clean_trc_data(trc_path)
        if trc_data is None:
            return {}
        
        # Use segmental method to calculate COM velocity
        try:
            vcm_data, time_vector, cm_df = self.compute_vcm_segmental(
                trc_data, SR=self.fs, SRd=self.fs, lowpass_cut=6.0)
        except Exception as e:
            print(f"Error computing segmental VCM for {data_label}: {e}")
            return {}
        
        # Use sync time data if available
        if sync_time_data is not None and len(sync_time_data) == len(vcm_data):
            time_vector = sync_time_data
        
        # Create individual output folder for this dataset (no subfolders)
        individual_output_folder = os.path.join(base_output_folder, folder_name, 'movement_analysis')
        os.makedirs(individual_output_folder, exist_ok=True)
        
        # Analyze transient segments
        results = self.analyze_transient_segments(vcm_data, time_vector, 
                                                 data_label, individual_output_folder)
        
        # Add VCM and COM data to results
        results['vcm_data'] = vcm_data
        results['time_vector'] = time_vector
        results['center_of_mass'] = cm_df
        
        # Save individual results directly to output folder (no subfolders)
        self.save_individual_results(results, individual_output_folder, data_label)
        
        return results
    
    def save_individual_results(self, results: Dict, output_folder: str, data_label: str):
        """
        Save individual results directly to output folder (no subfolders)
        """
        # Save pickle file directly to output folder
        pickle_path = os.path.join(output_folder, f'movement_results_{data_label}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Pickle results saved to: {pickle_path}")
        
        # Save CSV summary directly to output folder
        if 'metadata' in results:
            csv_data = []
            row = {
                'folder_name': data_label,
                'total_duration': results.get('metadata', {}).get('total_duration', np.nan),
                'sampling_rate': results.get('metadata', {}).get('sampling_rate', np.nan),
                'n_valley_points': results.get('metadata', {}).get('n_valley_points', np.nan),
                'n_gait_peaks': results.get('metadata', {}).get('n_gait_peaks', np.nan)
            }
            
            # Add time information for each stage
            for state_name in ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']:
                state_data = results.get(state_name, {})
                if state_data:
                    row[f'{state_name}_start'] = state_data.get('start_time', np.nan)
                    row[f'{state_name}_end'] = state_data.get('end_time', np.nan)
                    row[f'{state_name}_duration'] = state_data.get('duration', np.nan)
                    
                    if state_name == 'steady_gait':
                        row['steady_mean_velocity'] = state_data.get('mean_velocity', np.nan)
                        row['steady_velocity_std'] = state_data.get('velocity_std', np.nan)
                else:
                    row[f'{state_name}_start'] = np.nan
                    row[f'{state_name}_end'] = np.nan
                    row[f'{state_name}_duration'] = np.nan
            
            csv_data.append(row)
            df_individual = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_folder, f'movement_states_{data_label}.csv')
            df_individual.to_csv(csv_path, index=False)
            print(f"CSV results saved to: {csv_path}")
    
    def process_multiple_files(self, base_path: str, prefix: str = 'path2_', 
                              output_folder: str = None) -> Dict:
        """
        Batch process multiple files (using segmental weighted method)
        """
        # Load sync markers
        sync_map = self.load_sync_time_markers(base_path, prefix=prefix)
        
        # Find subdirectories
        subdirs = sorted([d for d in os.listdir(base_path) 
                         if d.startswith(prefix) and os.path.isdir(os.path.join(base_path, d))])
        
        if not subdirs:
            print(f"No directories found with prefix {prefix} in {base_path}")
            return {}
        
        all_results = {}
        
        for i, folder_name in enumerate(subdirs, start=1):
            print(f"Processing {i}/{len(subdirs)}: {folder_name}")
            
            try:
                # Find TRC files
                trc_pattern = os.path.join(base_path, folder_name, 'opensim', '*.trc')
                trc_files = glob.glob(trc_pattern)
                
                if not trc_files:
                    print(f"[WARN] No TRC files found for {folder_name}")
                    continue
                
                trc_path = trc_files[0]  # Use first TRC file
                
                # Get sync time data
                sync_entry = sync_map.get(folder_name, None)
                timeOfvideo = self.count_videotime_from_sync(sync_entry)
                
                # Process file
                results = self.process_single_file(trc_path, folder_name, timeOfvideo, folder_name, base_path)
                
                if results:
                    all_results[folder_name] = results
                    print(f"[OK] Successfully processed {folder_name}")
                    
                    # Print segmentation results summary
                    if 'sit_to_stand' in results and results['sit_to_stand']:
                        print(f"  - Sit→Stand: {results['sit_to_stand']['duration']:.2f}s")
                    if 'stand_to_walk' in results and results['stand_to_walk']:
                        print(f"  - Stand→Walk: {results['stand_to_walk']['duration']:.2f}s")
                    if 'steady_gait' in results and results['steady_gait']:
                        print(f"  - Steady Gait: {results['steady_gait']['duration']:.2f}s")
                    if 'walk_to_stop' in results and results['walk_to_stop']:
                        print(f"  - Walk→Stop: {results['walk_to_stop']['duration']:.2f}s")
                else:
                    print(f"[ERROR] Failed to process {folder_name}")
                    
            except Exception as e:
                print(f"[ERROR] Exception processing {folder_name}: {e}")
                continue
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {len(all_results)} files")
        print(f"Individual results saved in respective folders under: {base_path}")
        
        return all_results
    
    def save_results_summary(self, all_results: Dict, output_folder: str):
        """
        Save results summary to CSV file (deprecated - now using individual saves)
        """
        # This method is kept for compatibility but individual saves are preferred
        pass
    
    def process_dataset_transitions(self, rootFolder: str, prefix: str = 'path2_', 
                                   output_folder: str = None) -> Dict:
        """
        Complete dataset transient transition processing pipeline
        This is the main entry function that integrates all functionality
        """
        return self.process_multiple_files(rootFolder, prefix, output_folder)
    
    def analyze_single_trc(self, trc_path: str, output_folder: str = None, 
                          data_label: str = None) -> Dict:
        """
        Convenience function for analyzing single TRC file
        """
        if not os.path.exists(trc_path):
            raise FileNotFoundError(f"TRC file not found: {trc_path}")
        
        if data_label is None:
            data_label = os.path.splitext(os.path.basename(trc_path))[0]
        
        if output_folder is None:
            output_folder = os.path.dirname(trc_path)
        
        # For single file analysis, create folder structure
        folder_name = data_label
        individual_output_folder = os.path.join(output_folder, folder_name, 'movement_analysis')
        os.makedirs(individual_output_folder, exist_ok=True)
        
        return self.process_single_file(trc_path, folder_name, None, data_label, output_folder)


def main():
    """
    Main execution function - example usage
    """
    # Set paths - please modify according to your actual situation
    base_path = '/home/ntk/exo/ExoPulse/simulation/mocap_EMG_EEG_data/20241225'  # Data root directory
    
    # Create integrated segmentor
    segmentor = IntegratedMovementStateSegmentor(fs=30)
    
    print("=== Integrated Movement State Segmentor ===")
    print("Using segmental weighted method to calculate center of mass")
    print(f"Data path: {base_path}")
    print("Results will be saved directly in movement_analysis folders")
    
    # Method 1: Batch process entire dataset
    if os.path.exists(base_path):
        print("\nStarting batch processing...")
        results = segmentor.process_dataset_transitions(
            rootFolder=base_path, 
            prefix='path2_'
        )
        
        if results:
            print("\n=== Processing Results Summary ===")
            successful_count = len(results)
            print(f"Successfully processed files: {successful_count}")
            
            # Calculate detection success rate for each state
            state_counts = {'sit_to_stand': 0, 'stand_to_walk': 0, 
                           'steady_gait': 0, 'walk_to_stop': 0}
            
            for folder_name, result in results.items():
                for state_name in state_counts.keys():
                    if result.get(state_name) is not None:
                        state_counts[state_name] += 1
            
            print("Detection success rate for each state:")
            for state_name, count in state_counts.items():
                rate = count / successful_count * 100 if successful_count > 0 else 0
                print(f"  {state_name}: {count}/{successful_count} ({rate:.1f}%)")
                
            print("\nFile organization:")
            print("- PNG files: /path2_XX/movement_analysis/")
            print("- CSV files: /path2_XX/movement_analysis/")
            print("- Pickle files: /path2_XX/movement_analysis/")
    else:
        print(f"Data path does not exist: {base_path}")
        
        # Method 2: Process single file example
        single_trc_path = "path/to/your/single_file.trc"  # Please modify to actual path
        if os.path.exists(single_trc_path):
            print(f"\nProcessing single file: {single_trc_path}")
            result = segmentor.analyze_single_trc(single_trc_path)
            
            if result and 'metadata' in result:
                print("Single file processing results:")
                for state_name in ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']:
                    state_data = result.get(state_name)
                    if state_data:
                        print(f"  {state_name}: {state_data['start_time']:.2f}s - "
                              f"{state_data['end_time']:.2f}s (duration: {state_data['duration']:.2f}s)")
        else:
            print("Please modify the file paths in the program and run again")

    print("\nProgram execution completed!")


if __name__ == "__main__":
    main()
# %%