function emg_comprehensive_analysis
    % 1. 設定路徑與參數
    basePath = 'EMG Physical Action Data Set - del ';
    subs = {'sub1', 'sub2', 'sub3', 'sub4'};
    types = {'Normal', 'Aggressive'};
    fs = 4000;
    
    % 初始化儲存空間 (Table 格式)
    featureTable = table();
    
    fprintf('開始提取全域特徵...\n');
    
    % 2. 核心提取迴圈
    for t = 1:length(types)
        currentType = types{t};
        for s = 1:length(subs)
            currentSub = subs{s};
            folderPath = fullfile(basePath, currentSub, currentType, 'txt');
            files = dir(fullfile(folderPath, '*.txt'));
            
            for k = 1:length(files)
                try
                    data = load(fullfile(files(k).folder, files(k).name));
                    
                    % --- 預處理 ---
                    [b_bp, a_bp] = butter(4, [20 450]/(fs/2), 'bandpass');
                    data_filt = filtfilt(b_bp, a_bp, data);
                    
                    % --- 提取 4 種核心特徵 (對 8 個通道取平均) ---
                    f_rms = mean(rms(data_filt));
                    f_mav = mean(mean(abs(data_filt)));
                    f_zc  = mean(sum(abs(diff(data_filt > 0))) / size(data_filt, 1));
                    f_wl  = mean(sum(abs(diff(data_filt))));
                    
                    % --- 整合到 Table ---
                    newRow = table({currentSub}, {currentType}, f_rms, f_mav, f_zc, f_wl, ...
                        'VariableNames', {'Subject', 'Label', 'RMS', 'MAV', 'ZC', 'WL'});
                    featureTable = [featureTable; newRow];
                catch
                    continue;
                end
            end
        end
    end
    
    % 3. 視覺化分析 (正統方式：Boxplot)
    figure('Color', 'w', 'Name', 'Feature Statistical Analysis');
    featureList = {'RMS', 'MAV', 'ZC', 'WL'};
    
    for i = 1:4
        subplot(2, 2, i);
        boxplot(featureTable.(featureList{i}), featureTable.Label);
        title(['Distribution of ', featureList{i}]);
        grid on;
    end
    
    % 4. 輸出結果供後續機器學習使用
    fprintf('特徵提取完成。共有 %d 個樣本。\n', height(featureTable));
    assignin('base', 'emgFeatures', featureTable); % 將結果傳回到 MATLAB 主工作區
    disp('特徵矩陣已儲存至變數 emgFeatures。');
end