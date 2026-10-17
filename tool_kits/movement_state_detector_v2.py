# %%
"""
Integrated Movement State Segmentation - Fixed Version (for 2025 0617_0620 dataset)
Key improvements:
1. V1: Uses velocity gradient to find true movement onset
2. V2: Searches entire signal for most prominent transition valley
3. V3: Finds end valley after last high velocity point
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
    
    def __init__(self, fs: int = 30):
        self.fs = fs
        self.movement_states = {
            'sit_to_stand': [],
            'stand_to_walk': [],
            'steady_gait': [],
            'walk_to_stop': []
        }
        
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
        try:
            with open(trc_path, 'r') as f:
                header_lines = [next(f) for _ in range(4)]
            
            camera_info_header = header_lines[1].strip().split('\t')
            camera_info_values = header_lines[2].strip().split('\t')
            if len(camera_info_header) == len(camera_info_values):
                camera_info = dict(zip(camera_info_header, camera_info_values))
            else:
                camera_info = {"raw_header_lines": header_lines}
            
            header_line = header_lines[3].strip().split('\t')
            cleaned_header = [col for col in header_line if col != '']
            
            column_names = ['Frame#', 'Time'] + [f'{marker}_{axis}' 
                           for marker in cleaned_header[2:] for axis in ['X','Y','Z']]
            
            trc_data = pd.read_csv(trc_path, sep='\t', skiprows=5, header=None)
            trc_data.columns = column_names[:len(trc_data.columns)]
            
            trc_data = trc_data.apply(pd.to_numeric, errors='coerce')
            trc_data = trc_data.dropna(axis=1, how='all')
            
            return trc_data, camera_info
            
        except Exception as e:
            print(f"Error loading TRC file {trc_path}: {e}")
            return None, None
    
    def cal_seg_COM_trajectory(self, start_coords: np.ndarray, end_coords: np.ndarray, 
                              position_ratio: float, mass_ratio: float) -> np.ndarray:
        segment_COM = (start_coords + (end_coords - start_coords) * position_ratio) * mass_ratio
        return segment_COM
    
    def calculate_all_COMs(self, trc_data: pd.DataFrame, segments: dict = None) -> Dict[str, pd.DataFrame]:
        if segments is None:
            segments = self.default_segments
        
        num_timesteps = trc_data.shape[0]
        total_COM = np.zeros((num_timesteps, 3))
        segment_COMs = {}
        
        for segment, (start, end, position_ratio, mass_ratio) in segments.items():
            try:
                if isinstance(end, tuple):
                    coords0 = trc_data[[f"{end[0]}_X", f"{end[0]}_Y", f"{end[0]}_Z"]].values
                    coords1 = trc_data[[f"{end[1]}_X", f"{end[1]}_Y", f"{end[1]}_Z"]].values
                    end_coords = (coords0 + coords1) / 2.0
                else:
                    end_coords = trc_data[[f"{end}_X", f"{end}_Y", f"{end}_Z"]].values
                
                start_coords = trc_data[[f"{start}_X", f"{start}_Y", f"{start}_Z"]].values
                
                if start_coords.shape[0] != num_timesteps or end_coords.shape[0] != num_timesteps:
                    warnings.warn(f"Marker array length mismatch for segment {segment}")
                    continue
                
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
        
        segment_COMs['Total_COM'] = pd.DataFrame(total_COM, columns=['X','Y','Z'])
        return segment_COMs
    
    def compute_vcm_segmental(self, trc_data: pd.DataFrame, SR: int = 30, SRd: int = 30, 
                             lowpass_cut: float = 6.0, filter_order: int = 4, 
                             segments: dict = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        segs = self.calculate_all_COMs(trc_data, segments=segments)
        cm_raw = segs['Total_COM'].values
        
        N = cm_raw.shape[0]
        t_old = np.arange(N) / SR
        
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
            Nnew = int(round((N - 1) * SRd / SR)) + 1
            t = np.linspace(0, (N - 1) / SR, Nnew)
            cm = np.vstack([PchipInterpolator(t_old, cm_raw[:,i])(t) for i in range(3)]).T
            Fs_actual = SRd
        
        nyq = Fs_actual / 2.0
        if lowpass_cut >= nyq or lowpass_cut <= 0:
            cm_f = cm
        else:
            b, a = butter(filter_order, lowpass_cut / nyq)
            cm_f = np.column_stack([filtfilt(b, a, cm[:,i]) for i in range(3)])
        
        x = cm_f[:,0]
        z = cm_f[:,2]
        vx = np.gradient(x) * Fs_actual
        vz = np.gradient(z) * Fs_actual
        v = np.sqrt(vx**2 + vz**2)
        
        win_samples = max(1, int(round(0.05 * Fs_actual)))
        if win_samples > 1:
            v = pd.Series(v).rolling(window=win_samples, center=True, min_periods=1).mean().to_numpy()
        
        cm_df = pd.DataFrame(cm_f, columns=['X','Y','Z'])
        return v, t, cm_df
    
    def find_three_valleys(self, vcm_data: np.ndarray, time_vector: np.ndarray, 
                          y3: np.ndarray) -> List[int]:
        """
        Fixed valley detection:
        - V1: Uses velocity gradient to find movement onset (starting from 5s)
        - V2: Searches entire signal for best transition valley
        - V3: Finds valley after last high velocity
        """
        print("\n=== Finding 3 Key Valleys (Fixed Method) ===")
        
        # Step 1: Find movement onset using velocity gradient (starting after 5 seconds)
        start_search_idx = int(5.0 * self.fs)  # Start from 5 seconds
        if start_search_idx >= len(vcm_data):
            start_search_idx = 0  # Fallback if signal is too short
        
        vcm_gradient = np.gradient(vcm_data[start_search_idx:])
        window = max(3, int(0.2 * self.fs))
        vcm_gradient_smooth = pd.Series(vcm_gradient).rolling(
            window=window, center=True, min_periods=1).mean().to_numpy()
        
        gradient_threshold = np.std(vcm_gradient_smooth) * 0.5
        significant_change = np.where(np.abs(vcm_gradient_smooth) > gradient_threshold)[0]
        
        if len(significant_change) > 0:
            movement_onset_idx = start_search_idx + significant_change[0]
            print(f"Movement onset detected at: {time_vector[movement_onset_idx]:.2f}s")
        else:
            movement_onset_idx = start_search_idx + len(vcm_data[start_search_idx:]) // 4
        
        # Step 2: Find ALL candidate valleys
        inverted = -vcm_data
        all_valleys, valley_properties = find_peaks(
            inverted,
            distance=int(0.5 * self.fs),
            prominence=0.01,
            width=1
        )
        
        print(f"Found {len(all_valleys)} candidate valleys")
        
        if len(all_valleys) < 3:
            print("Not enough valleys, using fallback")
            return self._find_valleys_fallback(vcm_data, time_vector)
        
        # Step 3: Find V1 - valley near movement onset
        search_start_v1 = max(0, movement_onset_idx - int(3.0 * self.fs))
        search_end_v1 = movement_onset_idx + int(0.5 * self.fs)
        
        v1_candidates = all_valleys[(all_valleys >= search_start_v1) & (all_valleys <= search_end_v1)]
        
        if len(v1_candidates) > 0:
            distances_to_onset = np.abs(v1_candidates - movement_onset_idx)
            V1 = v1_candidates[np.argmin(distances_to_onset)]
        else:
            segment = vcm_data[search_start_v1:search_end_v1+1]
            V1 = search_start_v1 + np.argmin(segment)
        
        print(f"V1 at {time_vector[V1]:.2f}s (VCM={vcm_data[V1]:.3f})")
        
        # Step 4: Find V3 - valley after last high velocity
        last_high_velocity = len(vcm_data) - 1
        for i in range(len(vcm_data)-1, max(V1, len(vcm_data)//2), -1):
            if vcm_data[i] > 0.2:
                last_high_velocity = i
                break
        
        search_start_v3 = last_high_velocity
        search_end_v3 = len(vcm_data) - 1
        
        v3_candidates = all_valleys[(all_valleys >= search_start_v3) & (all_valleys <= search_end_v3)]
        
        if len(v3_candidates) > 0:
            v3_velocities = vcm_data[v3_candidates]
            V3 = v3_candidates[np.argmin(v3_velocities)]
        else:
            segment = vcm_data[search_start_v3:search_end_v3+1]
            V3 = search_start_v3 + np.argmin(segment)
        
        print(f"V3 at {time_vector[V3]:.2f}s (VCM={vcm_data[V3]:.3f})")
        
        # Step 5: Find V2 - transition valley between V1 and V3
        v2_candidates = all_valleys[(all_valleys > V1 + int(0.5*self.fs)) & 
                                   (all_valleys < V3 - int(1.0*self.fs))]
        
        if len(v2_candidates) > 0:
            v2_candidate_indices = [np.where(all_valleys == v)[0][0] for v in v2_candidates]
            v2_prominences = valley_properties['prominences'][v2_candidate_indices]
            v2_velocities = vcm_data[v2_candidates]
            
            v2_positions = (v2_candidates - V1) / (V3 - V1)
            position_weights = np.exp(-3 * (v2_positions - 0.3)**2)
            
            v2_scores = (1.0 / (v2_velocities + 0.01)) * v2_prominences * position_weights
            V2 = v2_candidates[np.argmax(v2_scores)]
            
            print(f"V2 at {time_vector[V2]:.2f}s (VCM={vcm_data[V2]:.3f})")
            print(f"  Prominence: {valley_properties['prominences'][np.where(all_valleys == V2)[0][0]]:.3f}")
        else:
            print("No V2 candidates, using fallback")
            search_range = vcm_data[V1:V3+1]
            V2 = V1 + np.argmin(search_range)
        
        valleys = [V1, V2, V3]
        
        if not (valleys[0] < valleys[1] < valleys[2]):
            print("ERROR: Valleys not in order")
            return self._find_valleys_fallback(vcm_data, time_vector)
        
        spacing_1_2 = (valleys[1] - valleys[0]) / self.fs
        spacing_2_3 = (valleys[2] - valleys[1]) / self.fs
        print(f"V1→V2: {spacing_1_2:.2f}s, V2→V3: {spacing_2_3:.2f}s")
        
        return valleys
    
    def _find_valleys_fallback(self, vcm_data: np.ndarray, time_vector: np.ndarray) -> List[int]:
        print("Using fallback valley detection...")
        
        movement_threshold = 0.1
        movement_mask = vcm_data > movement_threshold
        
        if not np.any(movement_mask):
            n = len(vcm_data)
            return [0, n//2, n-1]
        
        movement_indices = np.where(movement_mask)[0]
        movement_start = movement_indices[0]
        movement_end = movement_indices[-1]
        
        if movement_start > 0:
            V1 = np.argmin(vcm_data[:movement_start+1])
        else:
            V1 = 0
        
        if movement_end < len(vcm_data) - 1:
            V3 = movement_end + np.argmin(vcm_data[movement_end:])
        else:
            V3 = len(vcm_data) - 1
        
        search_end = movement_start + int((movement_end - movement_start) * 0.4)
        search_segment = vcm_data[movement_start:search_end+1]
        V2 = movement_start + np.argmin(search_segment)
        
        return [V1, V2, V3]
    
    def analyze_transient_segments(self, vcm_data: np.ndarray, time_vector: np.ndarray, 
                                 data_label: str = 'unknown', 
                                 output_folder: str = None) -> Dict:
        transient_results = {}
        vcm_data = vcm_data.flatten()
        time_vector = time_vector.flatten()
        
        if len(vcm_data) == 0:
            return {'T1': None, 'T2': None, 'T3': None, 'steady': None}
        
        if len(time_vector) != len(vcm_data):
            time_vector = np.arange(len(vcm_data)) / self.fs
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(time_vector, vcm_data, 'k-', linewidth=1.5, label='VCM (Segmental)')
        
        # Compute y3 for valley detection
        h1 = np.array([-1, -2, 0, 2, 1], dtype=float)
        y = convolve(vcm_data, h1, mode='same')
        y2 = y**2
        y2n = y2 if np.max(y2) == 0 else (y2 / np.max(y2))
        
        window_size = min(31, len(y2n))
        if window_size > 0:
            y3 = convolve(y2n, np.ones(window_size) / window_size, mode='same')
            if np.max(y3) != 0:
                y3 = y3 / np.max(y3)
        else:
            y3 = y2n
        
        # Find 3 valleys
        valley_locations = self.find_three_valleys(vcm_data, time_vector, y3)
        
        # Visualize valleys
        if len(valley_locations) == 3:
            V1, V2, V3 = valley_locations
            
            valley_styles = [
                {'color': 'red', 'marker': 'o', 'size': 12, 'label': 'V1 (Start)'},
                {'color': 'orange', 'marker': 's', 'size': 12, 'label': 'V2 (Transition)'},
                {'color': 'red', 'marker': 'o', 'size': 12, 'label': 'V3 (End)'}
            ]
            
            for i, (v_loc, style) in enumerate(zip(valley_locations, valley_styles)):
                ax.plot(time_vector[v_loc], vcm_data[v_loc], 
                       style['marker'], markersize=style['size'], 
                       markerfacecolor=style['color'], 
                       markeredgecolor='darkred' if i != 1 else 'darkorange',
                       markeredgewidth=2.5, label=style['label'], zorder=10)
                
                valley_name = ['V1', 'V2', 'V3'][i]
                label_text = f"{valley_name}\n{time_vector[v_loc]:.2f}s\n{vcm_data[v_loc]:.3f}m/s"
                
                # Adjust annotation position to avoid blocking the curve
                # V1 and V3 at bottom, V2 slightly to the side
                if i == 0:  # V1
                    xytext_offset = (-30, 25)
                elif i == 1:  # V2
                    xytext_offset = (30, -45)
                else:  # V3
                    xytext_offset = (-30, -45)
                
                ax.annotate(label_text, 
                           xy=(time_vector[v_loc], vcm_data[v_loc]),
                           xytext=xytext_offset, 
                           textcoords='offset points',
                           ha='center', fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', 
                                    facecolor='yellow', 
                                    edgecolor='darkred' if i != 1 else 'darkorange',
                                    linewidth=2, alpha=0.9),
                           arrowprops=dict(arrowstyle='->', 
                                         color='darkred' if i != 1 else 'darkorange', 
                                         lw=2))
        
        legend_labels = []
        
        # Segment if we have 3 valleys
        if len(valley_locations) == 3:
            V1, V2, V3 = valley_locations
            
            print(f"\n=== Analyzing V2-V3 segment ===")
            
            sub_vcm = vcm_data[V2:V3+1]
            
            if len(sub_vcm) > 0:
                vcm_range = np.max(sub_vcm) - np.min(sub_vcm)
                
                threshold_strategies = [
                    ('50% range', np.min(sub_vcm) + 0.5 * vcm_range),
                    ('60% percentile', np.percentile(sub_vcm, 60)),
                    ('mean + 0.2*std', np.mean(sub_vcm) + 0.2 * np.std(sub_vcm)),
                    ('70% percentile', np.percentile(sub_vcm, 70))
                ]
                
                best_peaks = None
                best_strategy = None
                
                for strategy_name, threshold in threshold_strategies:
                    min_distance = max(1, int(0.25 * self.fs))
                    peaks, properties = find_peaks(sub_vcm, height=threshold, distance=min_distance)
                    
                    if len(peaks) >= 2:
                        if best_peaks is None or len(peaks) > len(best_peaks):
                            best_peaks = peaks
                            best_strategy = strategy_name
                
                if best_peaks is not None and len(best_peaks) >= 2:
                    peaks = best_peaks
                    print(f"Using '{best_strategy}', found {len(peaks)} peaks")
                    
                    X1_peak = V2 + peaks[0]
                    X2_peak = V2 + peaks[-1]
                    
                    # Visualize only first and last peaks
                    ax.plot(time_vector[X1_peak], vcm_data[X1_peak], 
                           '^', markersize=12, markerfacecolor='limegreen', 
                           markeredgecolor='darkgreen', markeredgewidth=2.5,
                           label='X1 (First Peak)', zorder=10)
                    ax.plot(time_vector[X2_peak], vcm_data[X2_peak], 
                           '^', markersize=12, markerfacecolor='limegreen',
                           markeredgecolor='darkgreen', markeredgewidth=2.5,
                           label='X2 (Last Peak)', zorder=10)
                    
                    ax.annotate(f'X1\n{time_vector[X1_peak]:.2f}s', 
                               xy=(time_vector[X1_peak], vcm_data[X1_peak]),
                               xytext=(-20, 25), textcoords='offset points',
                               ha='center', fontsize=9, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='lightgreen',
                                        edgecolor='darkgreen',
                                        linewidth=2, alpha=0.9),
                               arrowprops=dict(arrowstyle='->', 
                                             color='darkgreen', lw=2))
                    
                    ax.annotate(f'X2\n{time_vector[X2_peak]:.2f}s', 
                               xy=(time_vector[X2_peak], vcm_data[X2_peak]),
                               xytext=(20, 25), textcoords='offset points',
                               ha='center', fontsize=9, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='lightgreen',
                                        edgecolor='darkgreen',
                                        linewidth=2, alpha=0.9),
                               arrowprops=dict(arrowstyle='->', 
                                             color='darkgreen', lw=2))
                    
                    # Define 4 states
                    T1_start, T1_end = V1, V2
                    T2_start, T2_end = V2, X1_peak
                    T4_start, T4_end = X1_peak, X2_peak
                    T3_start, T3_end = X2_peak, V3
                    
                    def clamp(idx):
                        return max(0, min(idx, len(time_vector)-1))
                    
                    T1_start, T1_end = clamp(T1_start), clamp(T1_end)
                    T2_start, T2_end = clamp(T2_start), clamp(T2_end)
                    T3_start, T3_end = clamp(T3_start), clamp(T3_end)
                    T4_start, T4_end = clamp(T4_start), clamp(T4_end)
                    
                    colors = {'sit_to_stand': 'blue', 'stand_to_walk': 'green', 
                             'steady_gait': 'cyan', 'walk_to_stop': 'magenta'}
                    
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
                    
                    print(f'\n=== Results ===')
                    print(f'Sit→Stand: {time_vector[T1_start]:.2f}-{time_vector[T1_end]:.2f}s')
                    print(f'Stand→Walk: {time_vector[T2_start]:.2f}-{time_vector[T2_end]:.2f}s')
                    print(f'Steady Gait: {time_vector[T4_start]:.2f}-{time_vector[T4_end]:.2f}s')
                    print(f'Walk→Stop: {time_vector[T3_start]:.2f}-{time_vector[T3_end]:.2f}s')
                    
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
                            'n_valley_points': 3,
                            'n_gait_peaks': len(peaks),
                            'method': 'fixed_3_valleys_gradient_based'
                        }
                    }
                    
                else:
                    warnings.warn("Not enough peaks in V2-V3")
                    transient_results = {'T1': None, 'T2': None, 'T3': None, 'steady': None}
            else:
                transient_results = {'T1': None, 'T2': None, 'T3': None, 'steady': None}
        else:
            transient_results = {'T1': None, 'T2': None, 'T3': None, 'steady': None}
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('VCM (m/s)', fontsize=12)
        ax.set_title(f'Segmental Center of Mass Velocity Analysis ({data_label})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        if len(legend_labels) > 0:
            state_colors = ['blue', 'green', 'cyan', 'magenta']
            for i, label in enumerate(legend_labels):
                handles.append(plt.Rectangle((0,0),1,1, fc=state_colors[i], alpha=0.3, ec='none'))
                labels.append(label)
        
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        
        if output_folder and os.path.exists(output_folder):
            save_path = os.path.join(output_folder, f'segmental_transitional_stages_{data_label}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {save_path}")
        
        plt.close()
        return transient_results
    
    def load_sync_time_markers(self, base_path: str, prefix: str = 'path2_') -> Dict:
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
        
    def count_videotime_from_sync(self, sync_entry) -> Optional[np.ndarray]:
        if sync_entry is None or len(sync_entry) < 1:
            return None
        try:
            frame_time_data = np.array(sync_entry[0]).astype(float).reshape(-1)
            return frame_time_data.T
        except Exception:
            return None
    
    def process_single_file(self, trc_path: str, folder_name: str, sync_time_data: np.ndarray = None, 
                          data_label: str = None, base_output_folder: str = None) -> Dict:
        if data_label is None:
            data_label = folder_name
        
        print(f"\n{'='*60}")
        print(f"Processing: {data_label}")
        print(f"{'='*60}")
        
        trc_data, camera_info = self.load_and_clean_trc_data(trc_path)
        if trc_data is None:
            return {}
        
        try:
            vcm_data, time_vector, cm_df = self.compute_vcm_segmental(
                trc_data, SR=self.fs, SRd=self.fs, lowpass_cut=6.0)
        except Exception as e:
            print(f"Error: {e}")
            return {}
        
        if sync_time_data is not None and len(sync_time_data) == len(vcm_data):
            time_vector = sync_time_data
        
        output_folder = os.path.join(base_output_folder, folder_name, 'movement_analysis')
        os.makedirs(output_folder, exist_ok=True)
        
        results = self.analyze_transient_segments(vcm_data, time_vector, data_label, output_folder)
        
        results['vcm_data'] = vcm_data
        results['time_vector'] = time_vector
        results['center_of_mass'] = cm_df
        
        self.save_individual_results(results, output_folder, data_label)
        
        return results
    
    def save_individual_results(self, results: Dict, output_folder: str, data_label: str):
        pickle_path = os.path.join(output_folder, f'movement_results_{data_label}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        if 'metadata' in results:
            row = {'folder_name': data_label}
            row['total_duration'] = results.get('metadata', {}).get('total_duration', np.nan)
            row['n_peaks'] = results.get('metadata', {}).get('n_gait_peaks', np.nan)
            
            for state in ['sit_to_stand', 'stand_to_walk', 'steady_gait', 'walk_to_stop']:
                data = results.get(state, {})
                if data:
                    row[f'{state}_start'] = data.get('start_time', np.nan)
                    row[f'{state}_end'] = data.get('end_time', np.nan)
                    row[f'{state}_duration'] = data.get('duration', np.nan)
            
            df = pd.DataFrame([row])
            csv_path = os.path.join(output_folder, f'summary_{data_label}.csv')
            df.to_csv(csv_path, index=False)
    
    def process_multiple_files(self, base_path: str, prefix: str = 'path2_') -> Dict:
        sync_map = self.load_sync_time_markers(base_path, prefix=prefix)
        
        subdirs = sorted([d for d in os.listdir(base_path) 
                         if d.startswith(prefix) and os.path.isdir(os.path.join(base_path, d))])
        
        if not subdirs:
            print(f"No directories found with prefix {prefix}")
            return {}
        
        all_results = {}
        success = 0
        fail = 0
        
        for i, folder_name in enumerate(subdirs, start=1):
            print(f"\n[{i}/{len(subdirs)}] {folder_name}")
            
            try:
                trc_pattern = os.path.join(base_path, folder_name, 'opensim', '*.trc')
                trc_files = glob.glob(trc_pattern)
                
                if not trc_files:
                    print("[SKIP] No TRC files")
                    fail += 1
                    continue
                
                trc_path = trc_files[0]
                sync_entry = sync_map.get(folder_name, None)
                timeOfvideo = self.count_videotime_from_sync(sync_entry)
                
                results = self.process_single_file(trc_path, folder_name, timeOfvideo, folder_name, base_path)
                
                if results and results.get('T1') is not None:
                    all_results[folder_name] = results
                    success += 1
                    print("[✓] Success")
                else:
                    fail += 1
                    print("[✗] Failed")
                    
            except Exception as e:
                print(f"[ERROR] {e}")
                fail += 1
        
        print(f"\n{'='*60}")
        print(f"Complete: {success} success, {fail} failed")
        print(f"{'='*60}")
        
        return all_results


def main():
    base_path = os.environ["RESEARCH_DATA_ROOT"]
    
    segmentor = IntegratedMovementStateSegmentor(fs=30)
    
    print("Movement State Detector - Fixed Version")
    print("Improvements:")
    print("1. V1 uses velocity gradient for movement onset")
    print("2. V2 searches entire signal for best transition valley")
    print("3. V3 finds valley after last high velocity")
    
    if os.path.exists(base_path):
        results = segmentor.process_multiple_files(base_path, prefix='path2_')
    else:
        print(f"Path not found: {base_path}")

if __name__ == "__main__":
    main()
