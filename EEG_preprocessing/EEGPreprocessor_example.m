

%% Example Usage of EEGPreprocessor

% Specify the folder containing your raw EEG CSV files
folder_path = getenv('RESEARCH_EEG_INPUT');
assert(isfolder(folder_path), 'Set RESEARCH_EEG_INPUT to your approved EEG CSV folder.');

% -------------------------------------------------------------------------
% STEP 1: Initialize the EEGPreprocessor object
%   This constructor scans the folder for files matching 'EEG_path2_*.csv',
%   sets up channel labels, event markers, filter settings, etc.
% -------------------------------------------------------------------------
eeg = EEGPreprocessor(folder_path);

% -------------------------------------------------------------------------
% STEP 2: Run the full preprocessing pipeline
%   This single call performs:
%     1. read_and_align   – extract the segment between your start/end events
%     2. reduce_mean      – remove per-channel DC offset
%     3. detrend          – remove linear trends
%     4. notch_filter     – 60 Hz notch to remove mains noise
%     5. bandpass_filter  – 0.5–70 Hz Butterworth
%     6. apply_ASR        – Artifact Subspace Reconstruction cleaning via EEGLAB
% -------------------------------------------------------------------------
eeg = eeg.read_and_align();
eeg = eeg.preprocess();

% -------------------------------------------------------------------------
% STEP 3: Quick visual checks at different stages
%   - Plot the detrended signal of file #3
%   - Plot the ASR-cleaned signal of file #10
% -------------------------------------------------------------------------
eeg.plot_check('detrend', 3);   % time×channels plot after detrending
eeg.plot_check('asr', 10);      % time×channels plot after ASR cleaning

% -------------------------------------------------------------------------
% STEP 4: Export ASR-cleaned data for manual ICA in EEGLAB
%   This call transposes each [time×channels] matrix to [channels×time]
%   and creates workspace variables Data1, Data2, … for EEGLAB import.
% -------------------------------------------------------------------------
eeg = eeg.export_ASR_for_ICA();

%% --- After completing ICA manually in EEGLAB ---

% Place all your EEGLAB .set files in this folder:
set_folder = getenv('RESEARCH_ICA_DIR');

% -------------------------------------------------------------------------
% STEP 5: Import ICA results back into the EEGPreprocessor object
%   This loads each Data*.set into eeg.EEG_data(i).ica, preserving other fields.
% -------------------------------------------------------------------------
eeg = eeg.import_ICA_results(set_folder);

% -------------------------------------------------------------------------
% STEP 6: Compute per-channel Z-score of the ICA-cleaned data
%   Outputs a time×channels matrix stored in eeg.EEG_data(i).zcore
% -------------------------------------------------------------------------
eeg = eeg.zscore();
eeg = eeg.save_zscore_time(getenv('RESEARCH_EEG_OUTPUT'));
% -------------------------------------------------------------------------
% STEP 7: Final visual checks
%   - Plot the ICA-reconstructed time series of file #1
%   - Plot the Z-scored ICA data of file #5
% -------------------------------------------------------------------------
eeg.plot_check('ica', 1);       % time×channels plot after ICA
eeg.plot_check('zscore', 5);     % time×channels plot after Z-scoring

