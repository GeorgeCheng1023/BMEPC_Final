function emg_feature_analysis
    % EMG_FEATURE_ANALYSIS - Feature Extraction and Class Separation Visualization
    % Extracts features (RMS, MAV, ZC, WL) and visualizes Normal vs Aggressive classes.
    % Shows how different features can separate the two classes of actions.

    % Main Figure
    f = figure('Name', 'EMG Feature Space Analysis', ...
               'NumberTitle', 'off', ...
               'Position', [100, 100, 1000, 700], ...
               'Color', 'w', ...
               'MenuBar', 'none', ...
               'ToolBar', 'figure');

    % --- UI Controls ---
    panelControl = uipanel(f, 'Position', [0, 0.85, 1, 0.15], ...
                           'BackgroundColor', [0.94 0.94 0.94], ...
                           'Title', 'Feature Selection');
    
    uicontrol(panelControl, 'Style', 'text', 'String', 'Subject:', ...
        'Position', [20, 40, 60, 20], 'HorizontalAlignment', 'right', 'BackgroundColor', [0.94 0.94 0.94]);
    comboSub = uicontrol(panelControl, 'Style', 'popupmenu', 'String', {'All', 'sub1', 'sub2', 'sub3', 'sub4'}, ...
        'Position', [90, 40, 80, 25]);

    uicontrol(panelControl, 'Style', 'text', 'String', 'X-Axis Feature:', ...
        'Position', [200, 40, 100, 20], 'HorizontalAlignment', 'right', 'BackgroundColor', [0.94 0.94 0.94]);
    comboX = uicontrol(panelControl, 'Style', 'popupmenu', 'String', {'RMS', 'MAV', 'ZC (Rate)', 'WL'}, ...
        'Position', [310, 40, 100, 25], 'Value', 1);

    uicontrol(panelControl, 'Style', 'text', 'String', 'Y-Axis Feature:', ...
        'Position', [420, 40, 100, 20], 'HorizontalAlignment', 'right', 'BackgroundColor', [0.94 0.94 0.94]);
    comboY = uicontrol(panelControl, 'Style', 'popupmenu', 'String', {'RMS', 'MAV', 'ZC (Rate)', 'WL'}, ...
        'Position', [530, 40, 100, 25], 'Value', 4);

    uicontrol(panelControl, 'Style', 'pushbutton', 'String', 'Extract & Visualize', ...
        'Position', [650, 35, 150, 40], 'Callback', @processData, ...
        'FontWeight', 'bold', 'BackgroundColor', [0.8 0.9 0.8]);

    % Plot Axes
    ax = axes('Parent', f, 'Position', [0.1, 0.1, 0.8, 0.7]);
    grid(ax, 'on');
    xlabel(ax, 'Feature X'); ylabel(ax, 'Feature Y');
    title(ax, 'Feature Space Separation');

    % --- Processing Function ---
    function processData(~, ~)
        % 1. Setup
        cla(ax);
        legend(ax, 'off');
        
        selectedSub = comboSub.String{comboSub.Value};
        featX_Name = comboX.String{comboX.Value};
        featY_Name = comboY.String{comboY.Value};
        
        if strcmp(selectedSub, 'All')
            subs = {'sub1', 'sub2', 'sub3', 'sub4'};
        else
            subs = {selectedSub};
        end
        
        types = {'Normal', 'Aggressive'};
        colors = {'b', 'r'}; % Blue for Normal, Red for Aggressive
        
        hWait = waitbar(0, 'Starting Feature Extraction...', 'WindowStyle', 'modal');
        
        hold(ax, 'on');
        
        % 2. Loop through data
        totalSteps = length(subs) * 2;
        currentStep = 0;
        
        for t = 1:2 % Types: Normal, Aggressive
            currentType = types{t};
            allX = [];
            allY = [];
            
            for s = 1:length(subs)
                currentSub = subs{s};
                currentStep = currentStep + 1;
                waitbar(currentStep/totalSteps, hWait, sprintf('Processing %s - %s...', currentSub, currentType));
                
                % Get file list
                folderPath = fullfile('EMG Physical Action Data Set', currentSub, currentType, 'txt');
                files = dir(fullfile(folderPath, '*.txt'));
                
                for k = 1:length(files)
                    filePath = fullfile(files(k).folder, files(k).name);
                    try
                        data = load(filePath);
                        
                        % Pre-processing (Filter)
                        fs = 4000;
                        [b_bp, a_bp] = butter(4, [20 450]/(fs/2), 'bandpass');
                        w_notch = [49 51] / (fs/2);
                        [b_notch, a_notch] = butter(2, w_notch, 'stop');
                        
                        % Filter all channels
                        for ch = 1:8
                            data(:, ch) = filtfilt(b_bp, a_bp, data(:, ch));
                            data(:, ch) = filtfilt(b_notch, a_notch, data(:, ch));
                        end
                        
                        % Feature Extraction 
                        % We calculate the feature for each channel, then take the MEAN across 8 channels
                        % to represent the "whole body" action as a single point in the plot.
                        
                        valX = extractFeature(data, featX_Name);
                        valY = extractFeature(data, featY_Name);
                        
                        allX = [allX; mean(valX)]; 
                        allY = [allY; mean(valY)];
                        
                    catch
                        continue;
                    end
                end
            end
            
            % Plot this class
            scatter(ax, allX, allY, 60, colors{t}, 'filled', 'DisplayName', currentType, ...
                'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.7);
        end
        
        hold(ax, 'off');
        xlabel(ax, ['Mean ' featX_Name]);
        ylabel(ax, ['Mean ' featY_Name]);
        title(ax, sprintf('Feature Separation: %s vs %s (%s)', featX_Name, featY_Name, selectedSub));
        legend(ax, 'show', 'Location', 'best');
        grid(ax, 'on');
        
        delete(hWait);
    end

    function featVal = extractFeature(signal, featName)
        % Signal is N x 8 matrix
        % Returns 1 x 8 vector of feature values
        
        switch featName
            case 'RMS'
                % Root Mean Square: Measure of power
                featVal = rms(signal);
                
            case 'MAV'
                % Mean Absolute Value: Measure of amplitude
                featVal = mean(abs(signal));
                
            case 'ZC (Rate)'
                % Zero Crossing Rate: Measure of frequency information
                % Threshold to avoid noise
                threshold = 0; 
                % Calculate changes in sign
                % diff(signal > 0) returns 1 or -1 at crossing, 0 otherwise
                featVal = sum(abs(diff(signal > threshold))) ./ size(signal, 1);
                
            case 'WL'
                % Waveform Length: Measure of signal complexity/amplitude
                % Sum of absolute differences between adjacent samples
                featVal = sum(abs(diff(signal)));
        end
    end
end
