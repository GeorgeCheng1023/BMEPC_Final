% Compare EMG Signals: Normal vs Aggressive
% This script compares EMG signals from a Normal action and an Aggressive action side-by-side.

% 1. Define file paths
% You can change these to compare different actions or subjects
normalFile = 'EMG Physical Action Data Set/sub1/Normal/txt/Walking.txt';
aggressiveFile = 'EMG Physical Action Data Set/sub1/Aggressive/txt/Punching.txt';

normalFullPath = fullfile(pwd, normalFile);
aggressiveFullPath = fullfile(pwd, aggressiveFile);

% 2. Load Data
if isfile(normalFullPath) && isfile(aggressiveFullPath)
    fprintf('Loading Normal data: %s\n', normalFile);
    dataNormal = load(normalFullPath);
    
    fprintf('Loading Aggressive data: %s\n', aggressiveFile);
    dataAggressive = load(aggressiveFullPath);
    
    % Check dimensions
    [samplesNormal, chansNormal] = size(dataNormal);
    [samplesAggressive, chansAggressive] = size(dataAggressive);
    
    if chansNormal ~= 8 || chansAggressive ~= 8
        warning('Expected 8 channels for both files.');
    end
    
    % 3. Setup Plotting
    f = figure('Name', 'Comparison: Normal vs Aggressive', 'NumberTitle', 'off', 'Color', 'w');
    f.Position = [50, 50, 1400, 900]; % Large figure
    
    channelLabels = {'Right Bicep', 'Right Tricep', 'Left Bicep', 'Left Tricep', ...
                     'Right Thigh', 'Right Hamstring', 'Left Thigh', 'Left Hamstring'};
                 
    tNormal = 1:samplesNormal;
    tAggressive = 1:samplesAggressive;
    
    % 4. Plot Side-by-Side
    for i = 1:8
        % Plot Normal Action (Left Column)
        subplot(8, 2, 2*i - 1);
        plot(tNormal, dataNormal(:, i), 'g'); % Green for Normal
        ylabel(channelLabels{i}, 'FontWeight', 'bold');
        grid on;
        axis tight;
        if i == 1
            title(['Normal: Walking'], 'FontSize', 12);
        end
        if i == 8
            xlabel('Sample Index');
        else
            set(gca, 'XTickLabel', []); % Hide x-labels for inner plots
        end
        
        % Plot Aggressive Action (Right Column)
        subplot(8, 2, 2*i);
        plot(tAggressive, dataAggressive(:, i), 'r'); % Red for Aggressive
        grid on;
        axis tight;
        if i == 1
            title(['Aggressive: Punching'], 'FontSize', 12);
        end
        if i == 8
            xlabel('Sample Index');
        else
            set(gca, 'XTickLabel', []);
        end
        
        % Link axes for easier comparison of amplitude within the row
        linkaxes([subplot(8, 2, 2*i - 1), subplot(8, 2, 2*i)], 'y');
    end
    
    sgtitle('EMG Signal Comparison: Normal vs Aggressive (Subject 1)', 'FontSize', 16, 'FontWeight', 'bold');
    
    fprintf('Comparison plot generated.\n');
    
else
    error('One or both files not found. Please check paths.');
end
