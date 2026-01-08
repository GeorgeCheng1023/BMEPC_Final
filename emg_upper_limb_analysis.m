function emg_upper_limb_analysis
    % 參數設定
    basePath = 'EMG Physical Action Data Set';
    subs = {'sub1', 'sub2', 'sub3', 'sub4'};
    types = {'Normal', 'Aggressive'};
    fs = 4000;
    targetChannels = 1:4; % 鎖定前 4 個通道 (上肢)

    featureTable = table();
    fprintf('正在分析上肢前 4 通道特徵...\n');

    for t = 1:length(types)
        currentType = types{t};
        for s = 1:length(subs)
            currentSub = subs{s};
            folderPath = fullfile(basePath, currentSub, currentType, 'txt');
            files = dir(fullfile(folderPath, '*.txt'));
            
            for k = 1:length(files)
                try
                    % 讀取資料 (N x 8)
                    rawData = load(fullfile(files(k).folder, files(k).name));
                    
                    % 只取前 4 個通道
                    data = rawData(:, targetChannels);
                    
                    % --- 預處理 (帶通濾波 20-450Hz) ---
                    [b, a] = butter(4, [20 450]/(fs/2), 'bandpass');
                    data_filt = filtfilt(b, a, data);
                    
                    % --- 提取特徵 (對這 4 個通道取平均，代表上肢整體的特徵數值) ---
                    f_rms = mean(rms(data_filt));
                    f_mav = mean(mean(abs(data_filt)));
                    f_zc  = mean(sum(abs(diff(data_filt > 0))) / size(data_filt, 1));
                    f_wl  = mean(sum(abs(diff(data_filt))));
                    
                    % 整合到 Table
                    newRow = table({currentSub}, {currentType}, f_rms, f_mav, f_zc, f_wl, ...
                        'VariableNames', {'Subject', 'Label', 'RMS', 'MAV', 'ZC', 'WL'});
                    featureTable = [featureTable; newRow];
                catch
                    continue;
                end
            end
        end
    end

    % 繪製箱形圖 (與你剛才看到的圖格式相同，但這是專屬上肢的數據)
    figure('Color', 'w', 'Name', 'Upper Limb Feature Analysis (CH 1-4)');
    feats = {'RMS', 'MAV', 'ZC', 'WL'};
    for i = 1:4
        subplot(2, 2, i);
        boxplot(featureTable.(feats{i}), featureTable.Label);
        title(['Upper Limb: ', feats{i}]);
        ylabel('Feature Value');
        grid on;
    end
    
    assignin('base', 'upperLimbFeatures', featureTable);
    fprintf('分析完成，數據已存至 upperLimbFeatures 變數。\n');
end
