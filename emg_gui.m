function emg_gui
    % EMG_GUI - Graphical User Interface for visualizing EMG Data
    % Allows selection of Subject, Action Type, and Action Name.
    
    % Create the main figure window
    f = figure('Name', 'EMG Data Visualizer', ...
               'NumberTitle', 'off', ...
               'Position', [100, 100, 1200, 800], ...
               'Color', 'w', ...
               'MenuBar', 'none', ...
               'ToolBar', 'figure');

    % --- Control Panel (Top) ---
    panelHeight = 0.1;
    panelControl = uipanel(f, 'Position', [0, 1-panelHeight, 1, panelHeight], ...
                           'BackgroundColor', [0.94 0.94 0.94]);

    % 1. Subject Selection
    uicontrol(panelControl, 'Style', 'text', 'String', 'Subject:', ...
        'Position', [20, 25, 60, 20], ...
        'HorizontalAlignment', 'right', ...
        'BackgroundColor', [0.94 0.94 0.94], 'FontSize', 10);
        
    comboSub = uicontrol(panelControl, 'Style', 'popupmenu', ...
        'String', {'sub1', 'sub2', 'sub3', 'sub4'}, ...
        'Position', [90, 25, 80, 25], 'FontSize', 10);

    % 2. Action Type Selection
    uicontrol(panelControl, 'Style', 'text', 'String', 'Type:', ...
        'Position', [190, 25, 40, 20], ...
        'HorizontalAlignment', 'right', ...
        'BackgroundColor', [0.94 0.94 0.94], 'FontSize', 10);
        
    comboType = uicontrol(panelControl, 'Style', 'popupmenu', ...
        'String', {'Normal', 'Aggressive'}, ...
        'Position', [240, 25, 100, 25], ...
        'Callback', @updateActionList, 'FontSize', 10);

    % 3. Action Name Selection
    uicontrol(panelControl, 'Style', 'text', 'String', 'Action:', ...
        'Position', [360, 25, 50, 20], ...
        'HorizontalAlignment', 'right', ...
        'BackgroundColor', [0.94 0.94 0.94], 'FontSize', 10);
        
    comboAction = uicontrol(panelControl, 'Style', 'popupmenu', ...
        'String', {}, ... % Populated dynamically
        'Position', [420, 25, 120, 25], 'FontSize', 10);

    % 4. Plot Button
    uicontrol(panelControl, 'Style', 'pushbutton', 'String', 'Plot EMG', ...
        'Position', [580, 22, 120, 30], ...
        'Callback', @plotData, ...
        'FontWeight', 'bold', 'FontSize', 11, ...
        'BackgroundColor', [0.8 0.9 1]);

    % --- Plotting Panel (Bottom) ---
    panelPlot = uipanel(f, 'Position', [0, 0, 1, 1-panelHeight], ...
                        'BackgroundColor', 'w', 'BorderType', 'none');

    % Define Lists
    normalActions = {'Bowing', 'Clapping', 'Handshaking', 'Hugging', 'Jumping', ...
                     'Running', 'Seating', 'Standing', 'Walking', 'Waving'};
    aggressiveActions = {'Elbowing', 'Frontkicking', 'Hamering', 'Headering', 'Kneeing', ...
                         'Pulling', 'Punching', 'Pushing', 'Sidekicking', 'Slapping'};
                     
    % Initialize the UI
    updateActionList();
    
    % --- Callback Functions ---

    function updateActionList(~, ~)
        % Updates the Action Name dropdown based on selected Type
        val = comboType.Value;
        if val == 1 % Normal
            comboAction.String = normalActions;
        else % Aggressive
            comboAction.String = aggressiveActions;
        end
        comboAction.Value = 1; % Reset selection to first item
    end

    function plotData(~, ~)
        % Main plotting logic
        
        % 1. Get User Selections
        subIdx = comboSub.Value;
        subjects = comboSub.String;
        subName = subjects{subIdx};
        
        typeIdx = comboType.Value;
        types = comboType.String;
        typeName = types{typeIdx};
        
        actionIdx = comboAction.Value;
        actions = comboAction.String;
        actionName = actions{actionIdx};
        
        % 2. Construct File Path
        % Structure: EMG Physical Action Data Set/subX/Type/txt/Action.txt
        relativePath = fullfile('EMG Physical Action Data Set', subName, typeName, 'txt', [actionName, '.txt']);
        fullPath = fullfile(pwd, relativePath);
        
        % 3. Validate File
        if ~isfile(fullPath)
            errordlg(['File not found: ', fullPath], 'File Error');
            return;
        end
        
        % 4. Load Data
        try
            data = load(fullPath);
        catch ME
            errordlg(['Error loading file: ', ME.message], 'Load Error');
            return;
        end
        
        [numSamples, numChannels] = size(data);
        
        if numChannels ~= 8
            warndlg(['Expected 8 channels, found ', num2str(numChannels)], 'Data Warning');
        end
        
        % 5. Plotting
        % Clear existing axes in the plot panel
        delete(allchild(panelPlot));
        
        t = 1:numSamples;
        channelLabels = {'Right Bicep', 'Right Tricep', 'Left Bicep', 'Left Tricep', ...
                         'Right Thigh', 'Right Hamstring', 'Left Thigh', 'Left Hamstring'};
        
        % Create 8 subplots
        for i = 1:8
            ax = subplot(4, 2, i, 'Parent', panelPlot);
            plot(ax, t, data(:, i), 'b');
            
            % Styling
            title(ax, channelLabels{i}, 'FontWeight', 'bold', 'FontSize', 9);
            grid(ax, 'on');
            axis(ax, 'tight');
            
            % Only show X label on bottom plots
            if i == 7 || i == 8
                xlabel(ax, 'Sample Index');
            end
            ylabel(ax, 'uV');
        end
        
        % Add a main title using a text object at the top of the panel
        % (sgtitle doesn't work easily with uipanel parent in older MATLAB versions, 
        % so we use a text annotation or just rely on the window title)
        f.Name = sprintf('EMG Data Visualizer - %s | %s | %s', subName, typeName, actionName);
    end

end
