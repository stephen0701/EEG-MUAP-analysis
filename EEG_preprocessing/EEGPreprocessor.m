

%%

classdef EEGPreprocessor
    % EEGPreprocessor: Batch reading, aligning, preprocessing, and visualizing EEG data in MATLAB
    properties
        channel_location_file = ''; % Set to the approved 32-channel .ced file
        folder_path          % Directory containing EEG CSV files
        files                % List of EEG CSV files matching EEG_path2_*.csv
        EEG_data             % Struct array storing raw and processed data for each file
        fs = 500;            % Sampling frequency in Hz
        ch_list              % Cell array of EEG channel labels
        n1 = 35;             % Primary event ID column index in raw data
        n2 = 38;             % Secondary event ID column index for fallback
        start_event = 130;   % Event code marking the start of alignment window
        end_event   = 141;   % Event code marking the end of alignment window
        columns_to_process   % Column indices for EEG channel data (3 to 34)
        asr_threshold = 10;  % Threshold parameter for ASR cleaning
    end

    methods
        function obj = EEGPreprocessor(folder_path)
            % Constructor: initialize file list, channel info, and preallocate data struct
            obj.folder_path = folder_path;
            obj.channel_location_file = getenv('RESEARCH_CHANNEL_LOCATIONS');
            % Gather all CSV files matching the naming pattern
            % ----- please change the filename of your .csv; ex:EMG_path2_*.csv ----- 
            obj.files = dir(fullfile(folder_path, 'EEG_path2_*.csv'));
            % Define which columns contain EEG channel data
            obj.columns_to_process = 3:34;
            % Define labels for each EEG channel
            obj.ch_list = { 'Fp1','Fp2','AF3','AF4','F7','F3','Fz','F4','F8', ...
                            'FT7','FC3','FCz','FC4','FT8','T7','C3','Cz', ...
                            'C4','T8','TP7','CP3','CPz','CP4','TP8', ...
                            'P7','P3','Pz','P4','P8','O1','Oz','O2' };
            % Preallocate EEG_data as an empty struct array of the same length
            obj.EEG_data = struct('filename', {}, 'original', {}, 'alignment', {}, ...
                              'time', {}, 'reducemean', {}, 'detrend', {}, ...
                              'notch', {}, 'bandpass', {}, 'asr', {}, ...
                              'asr_base', {}, 'asr_move', {});
            obj.EEG_data(numel(obj.files)).filename = [];
        end

        function obj = read_and_align(obj)
            % read_and_align: Load each CSV, locate event markers, and extract the aligned segment
            for i = 1:numel(obj.files)
                try
                    % Build full path and read table preserving variable names
                    fp = fullfile(obj.files(i).folder, obj.files(i).name);
                    T = readtable(fp, 'VariableNamingRule','preserve');
                    raw = table2array(T);

                    % Find start and end indices using primary event ID column
                    s = find(raw(:, obj.n1) == obj.start_event, 1);
                    e = find(raw(:, obj.n1) == obj.end_event, 1, 'last');
                    % If primary markers missing, use secondary column as fallback
                    if isempty(s) || isempty(e)
                        s = find(raw(:, obj.n2) == obj.start_event, 1);
                        e = find(raw(:, obj.n2) == obj.end_event, 1, 'last');
                    end

                    % Warn and skip file if no valid segment found
                    if isempty(s) || isempty(e)
                        warning('Cannot align file %s', obj.files(i).name);
                        continue;
                    end

                    % Extract the segment between the start and end markers
                    seg = raw(s:e, :);
                    time = seg(:, 1);

                    % Store raw and aligned data in the struct array
                    obj.EEG_data(i).filename = obj.files(i).name;
                    obj.EEG_data(i).original = raw;
                    obj.EEG_data(i).alignment = seg(:, obj.columns_to_process);
                    obj.EEG_data(i).time = time;
                catch ME
                    % Issue warning on read errors
                    warning('Error reading %s: %s', obj.files(i).name, ME.message);
                end
            end
        end

        function obj = preprocess(obj)
            % preprocess: Combined pipeline for mean removal, detrending,
            % notch filtering, bandpass filtering, and ASR cleaning
            % Requires EEGLAB on path for ASR functions
            eeglab('nogui'); % Initialize EEGLAB
            for i = 1:numel(obj.EEG_data)
                dataStruct = obj.EEG_data(i);
                if ~isempty(dataStruct.alignment)
                    
                    % 1) Remove DC offset per channel
                    data = dataStruct.alignment;
                    data = data - mean(data, 1);
                    obj.EEG_data(i).reducemean = data;

                    % 2) Remove linear trend
                    data = detrend(data);
                    obj.EEG_data(i).detrend = data;

                    % 3) Notch filter at 60 Hz
                    [b_n, a_n] = designNotchPeakIIR('CenterFrequency', 60/(obj.fs/2), ...
                                               'Bandwidth',       2/(obj.fs/2), ...
                                               'Response',        'notch');
                    data = filtfilt(b_n, a_n, data);
                    obj.EEG_data(i).notch = data;

                    % 4) Butterworth bandpass 0.5–70 Hz
                    [b_bp, a_bp] = butter(4, [0.5 70]/(obj.fs/2), 'bandpass');
                    data = filtfilt(b_bp, a_bp, data);
                    obj.EEG_data(i).bandpass = data;

                    % 5) Artifact Subspace Reconstruction (ASR)
                    %   a) Import to EEGLAB structure
                    eegData = data'; % channels × samples
                    EEG = pop_importdata('dataformat','array','nbchan',0, ...
                                        'data', eegData, 'srate', obj.fs, 'pnts',0, 'xmin',0);
                    %   b) Load the actual acquisition montage; do not substitute one.
                    assert(isfile(obj.channel_location_file), ...
                        'Set RESEARCH_CHANNEL_LOCATIONS or eeg.channel_location_file to the .ced file.');
                    EEG = pop_chanedit(EEG, 'load', {obj.channel_location_file,'filetype','autodetect'});

                    %   c) Calibrate ASR on first 10 seconds
                    base_pnts = min(10*obj.fs, size(EEG.data,2));
                    state     = asr_calibrate(EEG.data(:,1:base_pnts), obj.fs, obj.asr_threshold);

                    %   d) Clean the full data and re-reference
                    [cleaned, ~] = asr_process(EEG.data, obj.fs, state, 0.5, 0);
                    EEG.data     = cleaned;
                    EEG           = pop_reref(EEG, []);
                    fullData     = EEG.data';

                    %   e) Store ASR outputs
                    obj.EEG_data(i).asr      = fullData;
                    obj.EEG_data(i).asr_base = fullData(1:base_pnts, :);
                    obj.EEG_data(i).asr_move = fullData(base_pnts+1:end, :);
                end
            end
        end
 
        function obj = export_ASR_for_ICA(obj)
            % export_ASR_for_ICA: Export ASR-cleaned data for manual ICA processing
            %
            % For each dataset in obj.EEG_data:
            %   - Check for the presence of the 'asr' field.
            %   - Transpose the time×channels matrix to channels×time.
            %   - Assign the transposed matrix to a base workspace variable,
            %     named 'Data1', 'Data2', etc., to facilitate EEGLAB import.
            %
            % Usage:
            %   eeg = eeg.export_ASR_for_ICA();
            %
            nFiles = numel(obj.EEG_data);
            for i = 1:nFiles
                if isfield(obj.EEG_data(i), 'asr') && ~isempty(obj.EEG_data(i).asr)
                    % Transpose: from [time × channels] to [channels × time]
                    data = obj.EEG_data(i).asr';
                    varName = sprintf('Data%d', i);
                    % Assign to MATLAB base workspace
                    assignin('base', varName, data);
                    fprintf('Exported EEG_data(%d).asr → workspace variable ''%s''\n', i, varName);
                else
                    warning('EEG_data(%d) is missing ASR data; skipping export.', i);
                end
            end
        end

        function obj = import_ICA_results(obj, set_folder)
            % import_ICA_results: Batch-load ICA results from EEGLAB .set files
            %
            % Scans the specified folder for .set files. For each file:
            %   - Extracts the numeric index from the filename (e.g., 'Data3.set' → idx=3).
            %   - Loads the dataset into an EEGLAB EEG structure.
            %   - Stores it in obj.EEG_data(idx).ica without overwriting other fields.
            %
            % Usage:
            %   eeg = eeg.import_ICA_results('path/to/set_folder');
            %
            setFiles = dir(fullfile(set_folder, '*.set'));
            for k = 1:numel(setFiles)
                fn = setFiles(k).name;
                % Extract first numeric substring as index
                numStr = regexp(fn, '\d+', 'match', 'once');
                if ~isempty(numStr)
                    idx = str2double(numStr);
                    try
                        % Load .set file into EEGLAB structure
                        EEG = pop_loadset('filename', fullfile(setFiles(k).folder, fn));
                        % Store under the corresponding EEG_data element
                        obj.EEG_data(idx).ica = EEG;
                        fprintf('Loaded %s → EEG_data(%d).ica\n', fn, idx);
                    catch ME
                        warning('Failed to load %s: %s', fn, ME.message);
                    end
                else
                    warning('Filename %s lacks a numeric index; file skipped.', fn);
                end
            end
        end

        function obj = zscore(obj)
            % zscore: Compute Z-score of ICA-cleaned data per channel
            %
            % For each element in obj.EEG_data:
            %   - Check for EEG_data(i).ica.data (channels×samples).
            %   - Perform zscore along each row (channel dimension).
            %   - Transpose back to [time × channels] and store in .zcore.
            %
            % Usage:
            %   eeg = eeg.zscore();
            %
            for i = 1:numel(obj.EEG_data)
                if isfield(obj.EEG_data(i), 'ica') && isfield(obj.EEG_data(i).ica, 'data')
                    X = obj.EEG_data(i).ica.data;  % [channels × samples]
                    % Z-score each channel independently
                    Z = zscore(X, 0, 2);           % [channels × samples]
                    % Convert back to [time × channels]
                    obj.EEG_data(i).zscore = Z';
                else
                    warning('EEG_data(%d) missing ICA data; Z-scoring skipped.', i);
                end
            end
        end

        function plot_check(obj, stage, idx)
            % plot_check: Visualize a specific processing stage for a given file index
            % Supports stages: 'alignment','reducemean','detrend','notch',
            %                  'bandpass','asr','ica','zcore'
            % Inputs:
            %   stage - String: processing stage
            %   idx   - Index of the file (1 to numel(files))

            % Validate inputs
            if nargin < 3
                error('Usage: plot_check(stage, idx)');
            end
            if idx < 1 || idx > numel(obj.EEG_data)
                error('Index out of range (1 to %d)', numel(obj.EEG_data));
            end

            dataStruct = obj.EEG_data(idx);
            % Determine data matrix and time vector based on stage
            switch lower(stage)
                case 'ica'
                    if ~isfield(dataStruct, 'ica') || isempty(dataStruct.ica)
                        error('No ICA data for file %s', dataStruct.filename);
                    end
                    % EEG_data(idx).ica.data is channels×samples
                    data = dataStruct.ica.data';  % transpose to time×channels
                    time = (0:size(data,1)-1)/obj.fs;
                case 'zscore'
                    if ~isfield(dataStruct, 'zscore') || isempty(dataStruct.zscore)
                        error('No Z-score data for file %s', dataStruct.filename);
                    end
                    data = dataStruct.zscore;       % already time×channels
                    time = (0:size(data,1)-1)/obj.fs;
                otherwise
                    if ~isfield(dataStruct, stage) || isempty(dataStruct.(stage))
                        error('No data at stage %s for file %s', stage, dataStruct.filename);
                    end
                    data = dataStruct.(stage);
                    time = dataStruct.time;
            end

            nCh = numel(obj.ch_list);
            y_min = min(data(:));
            y_max = max(data(:));

            % Create figure
            figure;
            sgtitle(sprintf('File %d: %s - %s', idx, dataStruct.filename, stage), 'Interpreter', 'none');

            % Plot channels
            ax = gobjects(nCh, 1);
            for ch = 1:nCh
                ax(ch) = subplot(nCh, 1, ch);
                plot(time, data(:, ch));
                title(obj.ch_list{ch}, 'Interpreter', 'none');
                ylim([y_min y_max]);
                if ch == nCh
                    xlabel('Time (s)');
                end
            end
            linkaxes(ax, 'x');
        end

        function obj = save_zscore_time(obj, out_folder)
            % save_zscore_time: Save each EEG_data entry's zscore (time×channels) and time
            % as separate CSV files.
            %
            % Usage:
            %   eeg = eeg.save_zscore_time(getenv('RESEARCH_EEG_OUTPUT'));
            %
            % Parameters:
            %   out_folder - (string) Output directory. If omitted, obj.folder_path is used.

            % If output folder not provided, use object's folder_path
            if nargin < 2 || isempty(out_folder)
                out_folder = obj.folder_path;
            end
            % Create output folder if it does not exist
            if ~exist(out_folder, 'dir')
                mkdir(out_folder);
            end

            nFiles = numel(obj.EEG_data);
            for i = 1:nFiles
                try
                    % Check presence of both zscore and time fields
                    if isfield(obj.EEG_data(i), 'zscore') && ~isempty(obj.EEG_data(i).zscore) ...
                        && isfield(obj.EEG_data(i), 'time') && ~isempty(obj.EEG_data(i).time)

                        Z = obj.EEG_data(i).zscore;   % matrix: [time × channels]
                        t = obj.EEG_data(i).time;     % vector: [time × 1] or similar

                        % Ensure time vector is a column
                        if isrow(t)
                            t = t(:);
                        end

                        % Check length compatibility; if incompatible, trim to the smaller length
                        if size(Z,1) ~= numel(t)
                            warning('EEG_data(%d): zscore and time lengths differ (z:%d vs t:%d). Trimming to the smaller length.', ...
                                    i, size(Z,1), numel(t));
                            nmin = min(size(Z,1), numel(t));
                            Z = Z(1:nmin, :);
                            t = t(1:nmin);
                        end

                        % Base filename (use original filename without extension when available)
                        if isfield(obj.EEG_data(i), 'filename') && ~isempty(obj.EEG_data(i).filename)
                            [~, baseName] = fileparts(obj.EEG_data(i).filename);
                        else
                            baseName = sprintf('Data%d', i);
                        end

                        % --- Save zscore to CSV (Time as first column, then channels) ---
                        % Build a table and use ch_list as column names where possible
                        nCh = size(Z,2);
                        if numel(obj.ch_list) < nCh
                            chNames = cell(1, nCh);
                            for k = 1:nCh
                                if k <= numel(obj.ch_list)
                                    chNames{k} = obj.ch_list{k};
                                else
                                    chNames{k} = sprintf('Ch%d', k);
                                end
                            end
                        else
                            chNames = obj.ch_list(1:nCh);
                        end

                        % Create table with channel names and prepend Time column
                        T = array2table(Z, 'VariableNames', matlab.lang.makeValidName(chNames));
                        T = addvars(T, t, 'Before', 1, 'NewVariableNames', 'Timestamp'); % put time first

                        zfile = fullfile(out_folder, sprintf('%s_zscore.csv', baseName));
                        try
                            writetable(T, zfile);
                        catch
                            % If writetable fails, fall back to writematrix (no header)
                            warning('Failed to write %s with writetable; falling back to writematrix (no header).', zfile);
                            writematrix([t, Z], zfile);
                        end

                        fprintf('Saved: %s\n', zfile);
                    else
                        % Skip entries missing required data
                        fprintf('EEG_data(%d) missing zscore or time; skipped.\n', i);
                    end
                catch ME
                    % Report processing error for this entry and continue
                    warning('Error processing EEG_data(%d): %s', i, ME.message);
                end
            end
        end
    end
end
