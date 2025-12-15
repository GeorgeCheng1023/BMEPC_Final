function emg_decision_boundary_viz
    % EMG_DECISION_BOUNDARY_VIZ
    % Visualizes decision boundaries for SVM, KNN, and Ensemble classifiers
    % in a 2D feature space (Mean RMS vs Mean Zero Crossing).
    %
    % This visualization helps understand how each algorithm partitions the
    % feature space to classify actions as 'Normal' or 'Aggressive'.

    % 1. Load Data
    hWait = waitbar(0, 'Loading Data...', 'WindowStyle', 'modal');
    try
        [X_full, Y] = loadAndExtractFeatures(hWait);
    catch ME
        delete(hWait);
        errordlg(ME.message);
        return;
    end
    
    if isempty(X_full)
        delete(hWait);
        msgbox('No data found.', 'Warning');
        return;
    end

    % 2. Select 2 Features for Visualization
    % To visualize in 2D, we need to reduce dimensionality.
    % We will use Mean RMS (Amplitude) and Mean ZC (Frequency) across channels.
    % Indices in X_full: RMS (1-8), MAV (9-16), ZC (17-24), WL (25-32)
    
    % Calculate Mean across 8 channels for the two selected feature types
    X_2D = [mean(X_full(:, 1:8), 2), mean(X_full(:, 17:24), 2)];
    featureNames = {'Mean RMS (Amplitude)', 'Mean ZC (Frequency)'};
    
    % 3. Setup Visualization
    classifiers = {'SVM', 'KNN', 'Ensemble'};
    
    f = figure('Name', 'Decision Boundaries', 'Color', 'w', 'Position', [100, 100, 1400, 500]);
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, 'Decision Boundaries in 2D Feature Space', 'FontSize', 16, 'FontWeight', 'bold');

    % Create Grid for Prediction (The "Plane")
    % Add some padding to the range
    xMin = min(X_2D(:,1)); xMax = max(X_2D(:,1));
    yMin = min(X_2D(:,2)); yMax = max(X_2D(:,2));
    xPad = (xMax - xMin) * 0.1;
    yPad = (yMax - yMin) * 0.1;
    
    resolution = 200; % Grid density
    xRange = linspace(xMin - xPad, xMax + xPad, resolution);
    yRange = linspace(yMin - yPad, yMax + yPad, resolution);
    [xGrid, yGrid] = meshgrid(xRange, yRange);
    XGridFlat = [xGrid(:), yGrid(:)];

    waitbar(0.5, hWait, 'Training models and generating boundaries...');

    % Define colors
    % Class 1 (Aggressive): Red
    % Class 2 (Normal): Blue
    % Background colors (lighter versions)
    colorAggressive = [1, 0.8, 0.8]; % Light Red
    colorNormal = [0.8, 0.8, 1];     % Light Blue
    cmapRegion = [colorAggressive; colorNormal];

    for i = 1:length(classifiers)
        nexttile;
        name = classifiers{i};
        
        % Train Model on 2D data
        % Note: We train on the 2D projection specifically for this visualization
        % so the boundaries match the plot.
        mdl = trainModel(name, X_2D, Y);
        
        % Predict on Grid
        predictedLabels = predict(mdl, XGridFlat);
        
        % Convert predictions to numeric indices for plotting
        % 'Aggressive' -> 1, 'Normal' -> 2 (alphabetical order usually, but let's check)
        % We force a known order to match colors
        isNormal = strcmp(predictedLabels, 'Normal');
        predGridIdx = ones(size(predictedLabels)); % Default 1 (Aggressive)
        predGridIdx(isNormal) = 2;                 % Set 2 (Normal)
        
        predGrid = reshape(predGridIdx, size(xGrid));
        
        hold on;
        
        % Plot Decision Regions
        % We use imagesc to color the plane
        imagesc(xRange, yRange, predGrid);
        colormap(gca, cmapRegion);
        
        % Plot Actual Data Points
        % We need to separate data by class to plot with different markers/colors
        idxAgg = strcmp(Y, 'Aggressive');
        idxNorm = strcmp(Y, 'Normal');
        
        p1 = plot(X_2D(idxAgg, 1), X_2D(idxAgg, 2), 'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 6, 'DisplayName', 'Aggressive');
        p2 = plot(X_2D(idxNorm, 1), X_2D(idxNorm, 2), 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 6, 'DisplayName', 'Normal');
        
        % Formatting
        title(name, 'FontSize', 14, 'FontWeight', 'bold');
        xlabel(featureNames{1});
        ylabel(featureNames{2});
        axis xy; % Ensure Y axis is correct direction
        axis([xMin-xPad, xMax+xPad, yMin-yPad, yMax+yPad]);
        box on;
        
        if i == 1
            legend([p1, p2], 'Location', 'best');
        end
        hold off;
    end
    
    delete(hWait);
    
    % --- Helper: Train Model ---
    function mdl = trainModel(type, X, Y)
        switch type
            case 'SVM'
                % Using RBF kernel for non-linear boundaries which look better in 2D
                mdl = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'Standardize', true);
            case 'KNN'
                mdl = fitcknn(X, Y, 'NumNeighbors', 5, 'Standardize', true);
            case 'Ensemble'
                mdl = fitcensemble(X, Y, 'Method', 'Bag', 'NumLearningCycles', 50);
        end
    end

    % --- Helper: Load Data & Extract Features ---
    function [X, Y] = loadAndExtractFeatures(hWait)
        X = [];
        Y = {};
        
        subs = {'sub1', 'sub2', 'sub3', 'sub4'};
        types = {'Normal', 'Aggressive'};
        
        % Filter Design
        fs = 4000;
        [b_bp, a_bp] = butter(4, [20 450]/(fs/2), 'bandpass');
        w_notch = [49 51] / (fs/2);
        [b_notch, a_notch] = butter(2, w_notch, 'stop');
        
        totalSteps = length(subs) * length(types);
        step = 0;
        
        for s = 1:length(subs)
            for t = 1:length(types)
                step = step + 1;
                if isvalid(hWait)
                    waitbar(step/totalSteps * 0.5, hWait, sprintf('Loading %s - %s...', subs{s}, types{t}));
                end
                
                folderPath = fullfile('EMG Physical Action Data Set', subs{s}, types{t}, 'txt');
                files = dir(fullfile(folderPath, '*.txt'));
                
                for k = 1:length(files)
                    filePath = fullfile(files(k).folder, files(k).name);
                    try
                        raw = load(filePath);
                        if isempty(raw), continue; end
                        
                        % Filtering
                        filtered = zeros(size(raw));
                        for ch = 1:8
                            filtered(:, ch) = filtfilt(b_bp, a_bp, raw(:, ch));
                            filtered(:, ch) = filtfilt(b_notch, a_notch, filtered(:, ch));
                        end
                        
                        % Feature Extraction (32 features)
                        f_rms = rms(filtered);
                        f_mav = mean(abs(filtered));
                        threshold = 0;
                        f_zc = sum(abs(diff(filtered > threshold))) ./ size(filtered, 1);
                        f_wl = sum(abs(diff(filtered)));
                        
                        sampleFeatures = [f_rms, f_mav, f_zc, f_wl];
                        
                        X = [X; sampleFeatures];
                        Y = [Y; types{t}];
                        
                    catch
                        continue;
                    end
                end
            end
        end
    end
end
