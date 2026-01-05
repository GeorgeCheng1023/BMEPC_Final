function emg_upper_limb_analysis_2()
% =====================================================
% EMG Physical Action DataSet
% Upper-Limb Feature Extraction
% Normal vs Aggressive
% =====================================================

%% -------------------------
% Parameters
% -------------------------
basePath = 'EMG Physical Action Data Set - Del ';
subs = {'sub1', 'sub2', 'sub3', 'sub4'};
types = {'Normal', 'Aggressive'};

fs = 4000;
targetChannels = 1:4;

win_len = 250;
overlap = 125;
step = win_len - overlap;

featureTable = table();
fprintf('Analyzing upper-limb EMG features (ch1–ch4)...\n');

%% -------------------------
% Band-pass filter
% -------------------------
[b, a] = butter(4, [20 450] / (fs/2), 'bandpass');

%% -------------------------
% Feature names
% -------------------------
featureList = {'RMS', 'MAV', 'WL', 'VAR', 'ZC'};
featureNames = {};

for ch = targetChannels
    for k = 1:length(featureList)
        featureNames{end+1} = sprintf('Ch%d_%s', ch, featureList{k});
    end
end

%% -------------------------
% Main loop
% -------------------------
for t = 1:length(types)
    currentType = types{t};

    for s = 1:length(subs)
        currentSub = subs{s};

        % ⭐ 關鍵：照你成功版本的資料夾結構
        folderPath = fullfile(basePath, currentSub, currentType, 'txt');
        files = dir(fullfile(folderPath, '*.txt'));

        fprintf('Reading %s / %s : %d files\n', ...
            currentSub, currentType, length(files));

        for f = 1:length(files)
            try
                %% Load data
                rawData = load(fullfile(files(f).folder, files(f).name));

                emg = rawData(:, targetChannels);

                %% Preprocessing
                emg = emg - mean(emg);
                emg_filt = filtfilt(b, a, emg);

                %% Windowing
                N = size(emg_filt, 1);
                num_win = floor((N - win_len) / step) + 1;

                if num_win <= 0
                    continue;
                end

                %% Feature extraction
                for i = 1:num_win
                    idx = (i-1)*step + 1 : (i-1)*step + win_len;
                    window = emg_filt(idx, :);

                    feat = [];

                    for ch = 1:length(targetChannels)
                        x = window(:, ch);

                        RMS = sqrt(mean(x.^2));
                        MAV = mean(abs(x));
                        WL  = sum(abs(diff(x)));
                        VAR = var(x);
                        ZC  = sum(x(1:end-1) .* x(2:end) < 0);

                        feat = [feat RMS MAV WL VAR ZC];
                    end

                    %% Store
                    newRow = array2table(feat, ...
                        'VariableNames', featureNames);

                    newRow.Subject = {currentSub};
                    newRow.Type    = {currentType};
                    newRow.File    = {files(f).name};

                    featureTable = [featureTable; newRow];
                end

            catch
                warning('Failed to process %s', files(f).name);
                continue;
            end
        end
    end
end

fprintf('Feature extraction completed.\n');
fprintf('Total feature windows: %d\n', height(featureTable));

assignin('base', 'upperLimbFeatureTable', featureTable);

end

