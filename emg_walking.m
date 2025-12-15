% EMG Signal Graph Generator
% This script reads an EMG data file and plots the 8 channels.

% 1. Define the file path
% You can change this path to point to other files in the dataset
% e.g., 'EMG Physical Action Data Set/sub1/Aggressive/txt/Punching.txt'
relativePath = 'EMG Physical Action Data Set/sub3/Normal/txt/Walking.txt';
fullPath = fullfile(pwd, relativePath);

% 2. Load and Validate Data
if isfile(fullPath)
    fprintf('Loading data from: %s\n', fullPath);
    try
        data = load(fullPath);
    catch ME
        error('Failed to load data: %s', ME.message);
    end
    
    [numSamples, numChannels] = size(data);
    fprintf('Loaded %d samples with %d channels.\n', numSamples, numChannels);
    
    if numChannels ~= 8
        warning('Expected 8 channels as per readme, but found %d.', numChannels);
    end

    % 3. Plotting
    % Create a figure with white background
    f = figure('Name', 'EMG Signal Analysis', 'NumberTitle', 'off', 'Color', 'w');
    f.Position = [100, 100, 1200, 800]; % Make the figure larger

    % Define channel names (based on readme description of locations)
    % Mapping from readme:
    % Ch1: R-Bic (Right Bicep), Ch2: R-Tri (Right Tricep)
    % Ch3: L-Bic (Left Bicep),  Ch4: L-Tri (Left Tricep)
    % Ch5: R-Thi (Right Thigh), Ch6: R-Ham (Right Hamstring)
    % Ch7: L-Thi (Left Thigh),  Ch8: L-Ham (Left Hamstring)
    channelLabels = {'Right Bicep', 'Right Tricep', 'Left Bicep', 'Left Tricep', ...
                     'Right Thigh', 'Right Hamstring', 'Left Thigh', 'Left Hamstring'};

    t = 1:numSamples; % Sample index

    for i = 1:numChannels
        subplot(4, 2, i);
        plot(t, data(:, i), 'b'); % Plot in blue
        
        % Styling
        title(channelLabels{i}, 'FontWeight', 'bold');
        xlabel('Sample Index');
        ylabel('Amplitude (uV)'); % Assuming microvolts, typical for EMG
        axis tight; % Fit axis to data
        grid on;
        set(gca, 'FontSize', 8);
    end
    
    % Add a main title
    [~, fileName, ~] = fileparts(fullPath);
    sgtitle(['EMG Signals: ' fileName], 'FontSize', 14, 'FontWeight', 'bold');
    
    fprintf('Plot generated successfully.\n');
else
    error('File not found: %s\nPlease check the path and try again.', fullPath);
end
