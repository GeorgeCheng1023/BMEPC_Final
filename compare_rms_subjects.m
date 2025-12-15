% Compare RMS across Subjects for a Specific Action
% This script calculates the Root Mean Square (RMS) of the EMG signals
% for a specific movement across all 4 subjects and plots the comparison.

% 1. Configuration
% Change these to analyze a different movement
actionType = 'Normal'; % 'Normal' or 'Aggressive'
actionName = 'Clapping'; % e.g., 'Walking', 'Punching', 'Clapping'
fileName = [actionName, '.txt'];

subjects = {'sub1', 'sub2', 'sub3', 'sub4'};
numSubjects = length(subjects);
numChannels = 8;

% Initialize matrix to store RMS values: Rows = Channels, Cols = Subjects
rmsData = zeros(numChannels, numSubjects);

% Channel Labels
channelLabels = {'R-Bicep', 'R-Tricep', 'L-Bicep', 'L-Tricep', ...
                 'R-Thigh', 'R-Hamst', 'L-Thigh', 'L-Hamst'};

% 2. Process Data
fprintf('Calculating RMS for action: %s (%s)\n', actionName, actionType);

for i = 1:numSubjects
    subName = subjects{i};
    
    % Construct file path
    % Path format: EMG Physical Action Data Set/subX/Type/txt/Action.txt
    relativePath = fullfile('EMG Physical Action Data Set', subName, actionType, 'txt', fileName);
    fullPath = fullfile(pwd, relativePath);
    
    if isfile(fullPath)
        fprintf('  Processing %s...\n', subName);
        
        % Load data
        try
            rawData = load(fullPath);
            
            % Calculate RMS for each channel (column)
            % RMS = sqrt(mean(x^2))
            rmsValues = rms(rawData); 
            
            % Store in matrix (transpose if necessary to match 8x1)
            rmsData(:, i) = rmsValues(:);
            
        catch ME
            warning('Failed to load or process %s: %s', fullPath, ME.message);
            rmsData(:, i) = NaN; % Mark as missing
        end
    else
        warning('File not found: %s', fullPath);
        rmsData(:, i) = NaN;
    end
end

% 3. Visualization
f = figure('Name', ['RMS Comparison - ' actionName], 'NumberTitle', 'off', 'Color', 'w');
f.Position = [100, 100, 1000, 600];

% Create grouped bar chart
b = bar(rmsData);

% Styling
title(['RMS Amplitude Comparison for "' actionName '" Action'], 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Muscle Channel', 'FontSize', 12);
ylabel('RMS Amplitude (uV)', 'FontSize', 12);

% Set X-axis labels to muscle names
set(gca, 'XTick', 1:numChannels, 'XTickLabel', channelLabels);
xtickangle(45); % Rotate labels for better readability

% Legend for subjects
legend(subjects, 'Location', 'bestoutside');

grid on;

% Add values on top of bars (optional, can be crowded)
% for i = 1:numSubjects
%     xtips = b(i).XEndPoints;
%     ytips = b(i).YEndPoints;
%     labels = string(round(b(i).YData, 1));
%     text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
%         'VerticalAlignment', 'bottom', 'FontSize', 8);
% end

fprintf('RMS comparison plot generated.\n');
