function emg_upper_limb_analysis_3()



basePath = 'EMG Physical Action Data Set - Del';  
subs = {'sub1', 'sub2', 'sub3', 'sub4'};        
types = {'Normal', 'Aggressive'};              

fs = 4000;                                      
targetChannels = 1:4;                            
win_len = 250;                                   
overlap = 125;                                
step = win_len - overlap;

featureTable = table();                          
fprintf('Analysis...\n');


[b, a] = butter(4, [20 450] / (fs/2), 'bandpass');


for t = 1:length(types)
    currentType = types{t};
    for s = 1:length(subs)
        currentSub = subs{s};
        folderPath = fullfile(basePath, currentSub, currentType, 'txt'); 
        files = dir(fullfile(folderPath, '*.txt'));
        
        for k = 1:length(files)
            try
               
                rawData = load(fullfile(files(k).folder, files(k).name));
                
                
                data = rawData(:, targetChannels);
                
               
                data = data - mean(data);
                data_filt = filtfilt(b, a, data);
                
           
                f_rms = mean(sqrt(mean(data_filt.^2)));         % RMS
                f_mav = mean(mean(abs(data_filt)));             % MAV
                f_var = mean(var(data_filt));                   % VAR
                f_zc  = mean(sum(diff(data_filt > 0) ~= 0));   % ZC
                f_wl  = mean(sum(abs(diff(data_filt))));       % WL
                
               
                newRow = table({currentSub}, {currentType}, f_rms, f_mav, f_var, f_zc, f_wl, ...
                    'VariableNames', {'Subject', 'Label', 'RMS', 'MAV', 'VAR', 'ZC', 'WL'});
                
                featureTable = [featureTable; newRow];
            catch
                continue;
            end
        end
    end
end

% 將結果存到 base workspace
assignin('base', 'upperLimbFeatureTable', featureTable);

fprintf('特徵提取完成，數據已存至 upperLimbFeatureTable。\n');

%% -------------------------
% Boxplot 視覺化 (所有特徵在同一張 figure)
% -------------------------
feats = {'RMS', 'MAV', 'VAR', 'ZC', 'WL'};
colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980]; % Normal / Aggressive 顏色

figure('Color','w', 'Name', 'Upper Limb Features - Boxplot');
for i = 1:length(feats)
    subplot(2,3,i);
    
    % 先把資料拆成兩組
    normalData = featureTable.(feats{i})(strcmp(featureTable.Label,'Normal'));
    aggData    = featureTable.(feats{i})(strcmp(featureTable.Label,'Aggressive'));
    
    % 合併成矩陣給 boxplot
    dataForBox = [normalData; aggData];
    group = [repmat({'Normal'}, length(normalData), 1); repmat({'Aggressive'}, length(aggData), 1)];
    
    % 畫 boxplot
    h = boxplot(dataForBox, group, 'Colors', colors(1:2,:), 'Symbol', '');
    title(feats{i});
    ylabel('Feature Value');
    xlabel('Action Type');
    grid on;
    
    % 設定 box 顏色
    set(findobj(gca,'Tag','Box'),'LineWidth',1.5);
end
