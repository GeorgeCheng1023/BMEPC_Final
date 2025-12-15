function emg_spectral_comparison
    % EMG_SPECTRAL_COMPARISON - Compare Spectral Content of EMG Signals
    % Highlights frequency components of Aggressive vs Normal activities.
    % Showcases band-pass and notch filtering to remove artifacts.

    % Create the main figure window
    f = figure('Name', 'EMG Spectral Analysis: Normal vs Aggressive', ...
               'NumberTitle', 'off', ...
               'Position', [100, 100, 1000, 700], ...
               'Color', 'w', ...
               'MenuBar', 'none', ...
               'ToolBar', 'figure');

    % --- Control Panel (Top) ---
    panelHeight = 0.22;
    panelControl = uipanel(f, 'Position', [0, 1-panelHeight, 1, panelHeight], ...
                           'BackgroundColor', [0.94 0.94 0.94], ...
                           'Title', 'Comparison Settings');

    % Define Lists
    subjects = {'sub1', 'sub2', 'sub3', 'sub4'};
    types = {'Normal', 'Aggressive'};
    normalActions = {'Bowing', 'Clapping', 'Handshaking', 'Hugging', 'Jumping', ...
                     'Running', 'Seating', 'Standing', 'Walking', 'Waving'};
    aggressiveActions = {'Elbowing', 'Frontkicking', 'Hamering', 'Headering', 'Kneeing', ...
                         'Pulling', 'Punching', 'Pushing', 'Sidekicking', 'Slapping'};
    channelLabels = {'Right Bicep', 'Right Tricep', 'Left Bicep', 'Left Tricep', ...
                     'Right Thigh', 'Right Hamstring', 'Left Thigh', 'Left Hamstring'};

    % --- Set 1 Controls (Blue) ---
    uicontrol(panelControl, 'Style', 'text', 'String', 'Signal 1 (Blue):', ...
        'Position', [20, 100, 100, 20], 'HorizontalAlignment', 'left', 'FontWeight', 'bold', 'ForegroundColor', 'b');

    uicontrol(panelControl, 'Style', 'text', 'String', 'Subject:', 'Position', [20, 75, 50, 20], 'HorizontalAlignment', 'right');
    comboSub1 = uicontrol(panelControl, 'Style', 'popupmenu', 'String', subjects, ...
        'Position', [75, 75, 70, 25]);

    uicontrol(panelControl, 'Style', 'text', 'String', 'Type:', 'Position', [150, 75, 40, 20], 'HorizontalAlignment', 'right');
    comboType1 = uicontrol(panelControl, 'Style', 'popupmenu', 'String', types, ...
        'Position', [195, 75, 90, 25], 'Callback', {@updateActionList, 1});

    uicontrol(panelControl, 'Style', 'text', 'String', 'Action:', 'Position', [290, 75, 40, 20], 'HorizontalAlignment', 'right');
    comboAction1 = uicontrol(panelControl, 'Style', 'popupmenu', 'String', normalActions, ...
        'Position', [335, 75, 100, 25]);

    % --- Set 2 Controls (Red) ---
    uicontrol(panelControl, 'Style', 'text', 'String', 'Signal 2 (Red):', ...
        'Position', [20, 40, 100, 20], 'HorizontalAlignment', 'left', 'FontWeight', 'bold', 'ForegroundColor', 'r');

    uicontrol(panelControl, 'Style', 'text', 'String', 'Subject:', 'Position', [20, 15, 50, 20], 'HorizontalAlignment', 'right');
    comboSub2 = uicontrol(panelControl, 'Style', 'popupmenu', 'String', subjects, ...
        'Position', [75, 15, 70, 25]);

    uicontrol(panelControl, 'Style', 'text', 'String', 'Type:', 'Position', [150, 15, 40, 20], 'HorizontalAlignment', 'right');
    comboType2 = uicontrol(panelControl, 'Style', 'popupmenu', 'String', types, ...
        'Position', [195, 15, 90, 25], 'Value', 2, 'Callback', {@updateActionList, 2});

    uicontrol(panelControl, 'Style', 'text', 'String', 'Action:', 'Position', [290, 15, 40, 20], 'HorizontalAlignment', 'right');
    comboAction2 = uicontrol(panelControl, 'Style', 'popupmenu', 'String', aggressiveActions, ...
        'Position', [335, 15, 100, 25]);

    % --- Common Controls ---
    uicontrol(panelControl, 'Style', 'text', 'String', 'Channel to Analyze:', ...
        'Position', [500, 80, 120, 20], 'HorizontalAlignment', 'left', 'FontWeight', 'bold');
    comboChannel = uicontrol(panelControl, 'Style', 'popupmenu', 'String', channelLabels, ...
        'Position', [500, 55, 150, 25]);

    chkFilter = uicontrol(panelControl, 'Style', 'checkbox', 'String', 'Apply Filters (BP + Notch)', ...
        'Position', [500, 25, 180, 25], 'Value', 1, 'FontWeight', 'bold');

    uicontrol(panelControl, 'Style', 'pushbutton', 'String', 'Compare Signals', ...
        'Position', [700, 30, 150, 50], ...
        'Callback', @plotComparison, ...
        'FontWeight', 'bold', 'FontSize', 11, ...
        'BackgroundColor', [0.8 0.9 0.8]);

    % --- Plotting Panel (Bottom) ---
    panelPlot = uipanel(f, 'Position', [0, 0, 1, 1-panelHeight], ...
                        'BackgroundColor', 'w', 'BorderType', 'none');

    % Initialize UI
    % (Set 2 is already set to Aggressive/AggressiveActions by default creation, but let's ensure consistency)
    
    % --- Callback Functions ---

    function updateActionList(~, ~, setNum)
        % Updates the Action Name dropdown based on selected Type for Set 1 or 2
        if setNum == 1
            cType = comboType1;
            cAction = comboAction1;
        else
            cType = comboType2;
            cAction = comboAction2;
        end
        
        val = cType.Value;
        if val == 1 % Normal
            cAction.String = normalActions;
        else % Aggressive
            cAction.String = aggressiveActions;
        end
        cAction.Value = 1;
    end

    function plotComparison(~, ~)
        % Main plotting logic
        
        % Create a waitbar to show progress
        hWait = waitbar(0, 'Initializing...', 'Name', 'Processing Data', 'WindowStyle', 'modal');
        
        try
            % 1. Get Data for Signal 1
            waitbar(0.2, hWait, 'Loading and Filtering Signal 1...');
            [t1, y1, fs, name1] = loadAndProcess(comboSub1, comboType1, comboAction1);
            if isempty(t1)
                delete(hWait); 
                return; 
            end
            
            % 2. Get Data for Signal 2
            waitbar(0.5, hWait, 'Loading and Filtering Signal 2...');
            [t2, y2, ~, name2] = loadAndProcess(comboSub2, comboType2, comboAction2);
            if isempty(t2)
                delete(hWait); 
                return; 
            end
            
            % 3. Plotting
            waitbar(0.7, hWait, 'Plotting Time Domain...');
            delete(allchild(panelPlot));
            
            % -- Time Domain Plot --
            axTime = subplot(2, 1, 1, 'Parent', panelPlot);
            hold(axTime, 'on');
            plot(axTime, t1, y1, 'b', 'DisplayName', name1);
            plot(axTime, t2, y2, 'r', 'DisplayName', name2);
            hold(axTime, 'off');
            
            title(axTime, ['Time Domain Signal - ' channelLabels{comboChannel.Value}], 'FontSize', 11, 'FontWeight', 'bold');
            xlabel(axTime, 'Time (s)');
            ylabel(axTime, 'Amplitude (uV)');
            legend(axTime, 'show', 'Location', 'best');
            grid(axTime, 'on');
            axis(axTime, 'tight');
            
            % -- Frequency Domain Plot (PSD) --
            waitbar(0.85, hWait, 'Calculating PSD...');
            axFreq = subplot(2, 1, 2, 'Parent', panelPlot);
            hold(axFreq, 'on');
            
            % Calculate PSD using Welch's method
            % Window size: 512 samples, 50% overlap
            window = 512;
            noverlap = 256;
            nfft = 1024;
            
            [pxx1, f1] = pwelch(y1, window, noverlap, nfft, fs);
            [pxx2, f2] = pwelch(y2, window, noverlap, nfft, fs);
            
            plot(axFreq, f1, 10*log10(pxx1), 'b', 'LineWidth', 1.5, 'DisplayName', name1);
            plot(axFreq, f2, 10*log10(pxx2), 'r', 'LineWidth', 1.5, 'DisplayName', name2);
            hold(axFreq, 'off');
            
            title(axFreq, 'Power Spectral Density (PSD)', 'FontSize', 11, 'FontWeight', 'bold');
            xlabel(axFreq, 'Frequency (Hz)');
            ylabel(axFreq, 'Power/Frequency (dB/Hz)');
            legend(axFreq, 'show', 'Location', 'best');
            grid(axFreq, 'on');
            xlim(axFreq, [0, 500]); % Focus on 0-500 Hz range where EMG power lies
            
            waitbar(1, hWait, 'Done!');
            pause(0.5); % Short pause to let user see "Done"
            
        catch ME
            errordlg(['An error occurred: ' ME.message], 'Error');
        end
        
        % Clean up waitbar
        if isvalid(hWait)
            delete(hWait);
        end
    end

    function [t, y, fs, labelName] = loadAndProcess(cSub, cType, cAction)
        t = []; y = []; fs = 4000; labelName = '';
        
        subName = cSub.String{cSub.Value};
        typeName = cType.String{cType.Value};
        actionName = cAction.String{cAction.Value};
        labelName = sprintf('%s - %s (%s)', typeName, actionName, subName);
        
        % Construct Path
        relativePath = fullfile('EMG Physical Action Data Set', subName, typeName, 'txt', [actionName, '.txt']);
        fullPath = fullfile(pwd, relativePath);
        
        if ~isfile(fullPath)
            errordlg(['File not found: ', fullPath], 'File Error');
            return;
        end
        
        try
            data = load(fullPath);
        catch ME
            errordlg(['Error loading file: ', ME.message], 'Load Error');
            return;
        end
        
        % Select Channel
        chanIdx = comboChannel.Value;
        if size(data, 2) >= chanIdx
            raw_signal = data(:, chanIdx);
        else
            warndlg('Channel index out of range.', 'Error');
            return;
        end
        
        % Apply Filters if checked
        if chkFilter.Value
            % Bandpass (20-450 Hz)
            [b_bp, a_bp] = butter(4, [20 450]/(fs/2), 'bandpass');
            
            % Notch (50 Hz) - Using Bandstop filter
            % Replaces iirnotch with a standard Butterworth bandstop
            w_notch = [49 51] / (fs/2);
            [b_notch, a_notch] = butter(2, w_notch, 'stop');
            
            filtered_signal = filtfilt(b_bp, a_bp, raw_signal);
            filtered_signal = filtfilt(b_notch, a_notch, filtered_signal);
            y = filtered_signal;
        else
            y = raw_signal;
        end
        
        % Time vector
        t = (0:length(y)-1) / fs;
    end

end
