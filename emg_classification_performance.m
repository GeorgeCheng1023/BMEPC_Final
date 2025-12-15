function emg_classification_performance
    % EMG_CLASSIFICATION_PERFORMANCE
    % Trains classifiers to distinguish between Normal and Aggressive actions.
    % Compares performance of SVM, KNN, and Ensemble classifiers.
    % Presents training progress (Learning Curves) and validation accuracy.

    % Create Main Figure
    f = figure('Name', 'EMG Classification Performance', ...
               'NumberTitle', 'off', ...
               'Position', [100, 100, 1200, 700], ...
               'Color', 'w');
    
    % Create Tab Group
    tgroup = uitabgroup(f, 'Position', [0, 0.1, 1, 0.9]);
    tab1 = uitab(tgroup, 'Title', 'Model Comparison');
    tab2 = uitab(tgroup, 'Title', 'Confusion Matrices');
    tab3 = uitab(tgroup, 'Title', 'Learning Curves');
    
    % Axes for Tab 1 (Bar Chart)
    axBar = axes('Parent', tab1, 'Position', [0.15, 0.15, 0.7, 0.7]);
    title(axBar, 'Press "Run Analysis" to start...');
    
    % Tab 2 will be populated dynamically with tiledlayout
    
    % Axes for Tab 3 (Learning Curve)
    axLearn = axes('Parent', tab3, 'Position', [0.1, 0.1, 0.8, 0.8]);
    grid(axLearn, 'on');
    xlabel(axLearn, 'Training Set Size (%)');
    ylabel(axLearn, 'Validation Accuracy (%)');
    title(axLearn, 'Learning Curves');
    
    % Control Panel
    uicontrol(f, 'Style', 'pushbutton', 'String', 'Run Analysis', ...
        'Position', [500, 10, 200, 40], ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.2, 0.6, 0.2], 'ForegroundColor', 'w', ...
        'Callback', @runAnalysis);
        
    % --- Main Analysis Function ---
    function runAnalysis(~, ~)
        % 1. Load Data
        hWait = waitbar(0, 'Loading and Processing Data...', 'WindowStyle', 'modal');
        try
            [X, Y] = loadAllData(hWait);
        catch ME
            delete(hWait);
            errordlg(['Error loading data: ' ME.message], 'Error');
            return;
        end
        
        if isempty(X)
            delete(hWait);
            msgbox('No data found.', 'Warning');
            return;
        end
        
        % 2. Define Classifiers
        classifiers = {'SVM', 'KNN', 'Ensemble'};
        accuracies = zeros(1, length(classifiers));
        
        % 3. Cross-Validation Evaluation
        k = 5;
        cv = cvpartition(Y, 'KFold', k);
        
        waitbar(0.5, hWait, 'Training Classifiers...');
        fprintf('--- Starting Cross-Validation Evaluation ---\n');
        
        for i = 1:length(classifiers)
            accSum = 0;
            fprintf('Training Classifier: %s\n', classifiers{i});
            for j = 1:k
                fprintf('  Fold %d/%d...\n', j, k);
                trainIdx = training(cv, j);
                testIdx = test(cv, j);
                
                XTrain = X(trainIdx, :);
                YTrain = Y(trainIdx);
                XTest = X(testIdx, :);
                YTest = Y(testIdx);
                
                mdl = trainModel(classifiers{i}, XTrain, YTrain);
                YPred = predict(mdl, XTest);
                
                accSum = accSum + sum(strcmp(YPred, YTest)) / length(YTest);
            end
            accuracies(i) = (accSum / k) * 100;
            fprintf('  Accuracy for %s: %.2f%%\n', classifiers{i}, accuracies(i));
        end
        
        % 4. Update Bar Chart
        bar(axBar, accuracies, 'FaceColor', [0.2 0.4 0.6]);
        set(axBar, 'XTickLabel', classifiers, 'FontSize', 12);
        ylabel(axBar, 'Validation Accuracy (%)');
        title(axBar, sprintf('Classifier Performance (%d-Fold CV)', k));
        grid(axBar, 'on');
        ylim(axBar, [0 100]);
        
        for i = 1:length(accuracies)
            text(axBar, i, accuracies(i) + 2, sprintf('%.1f%%', accuracies(i)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
        end
        
        % 5. Confusion Matrices for All Models
        [~, bestIdx] = max(accuracies);
        bestName = classifiers{bestIdx};
        waitbar(0.7, hWait, 'Generating Confusion Matrices...');
        
        % Train/Test Split for Visualization
        cvSplit = cvpartition(Y, 'HoldOut', 0.3);
        XTrain = X(training(cvSplit), :);
        YTrain = Y(training(cvSplit));
        XTest = X(test(cvSplit), :);
        YTest = Y(test(cvSplit));
        
        % Clear previous content in tab2
        delete(tab2.Children);
        
        % Create layout for multiple charts
        t = tiledlayout(tab2, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        title(t, 'Confusion Matrices for All Models', 'FontSize', 14, 'FontWeight', 'bold');

        for i = 1:length(classifiers)
            modelName = classifiers{i};
            mdl = trainModel(modelName, XTrain, YTrain);
            YPred = predict(mdl, XTest);
            
            nexttile(t);
            confusionchart(YTest, YPred, 'Title', modelName);
        end
        
        % 6. Learning Curves
        waitbar(0.8, hWait, 'Generating Learning Curves...');
        fprintf('--- Generating Learning Curves ---\n');
        cla(axLearn);
        hold(axLearn, 'on');
        colors = {'r', 'g', 'b'};
        markers = {'o', 's', '^'};
        
        trainSizes = [0.2, 0.4, 0.6, 0.8]; % Fractions of data to use for training
        
        for i = 1:length(classifiers)
            fprintf('Processing Classifier: %s\n', classifiers{i});
            curveAcc = zeros(1, length(trainSizes));
            for t = 1:length(trainSizes)
                % Use a subset of data
                subsetSize = floor(trainSizes(t) * size(X, 1));
                fprintf('  Training Size: %.0f%% (%d samples)\n', trainSizes(t)*100, subsetSize);
                % Randomly select subset
                randIdx = randperm(size(X, 1), subsetSize);
                XSub = X(randIdx, :);
                YSub = Y(randIdx);
                
                % 5-fold CV on this subset
                if subsetSize < 10 % Too small
                    curveAcc(t) = 0; 
                    fprintf('    Skipping (too small)\n');
                    continue;
                end
                
                try
                    cvSub = cvpartition(YSub, 'KFold', 5);
                    subAccSum = 0;
                    for f = 1:5
                        trIdx = training(cvSub, f);
                        teIdx = test(cvSub, f);
                        mdl = trainModel(classifiers{i}, XSub(trIdx, :), YSub(trIdx));
                        pred = predict(mdl, XSub(teIdx, :));
                        subAccSum = subAccSum + sum(strcmp(pred, YSub(teIdx))) / length(YSub(teIdx));
                    end
                    curveAcc(t) = (subAccSum / 5) * 100;
                    fprintf('    Accuracy: %.2f%%\n', curveAcc(t));
                catch ME
                    curveAcc(t) = NaN;
                    fprintf('    Error: %s\n', ME.message);
                end
            end
            plot(axLearn, trainSizes*100, curveAcc, [colors{i} '-' markers{i}], ...
                'LineWidth', 2, 'DisplayName', classifiers{i});
        end
        
        legend(axLearn, 'show', 'Location', 'southeast');
        hold(axLearn, 'off');
        
        waitbar(1, hWait, 'Done!');
        pause(1);
        delete(hWait);
        
        msgbox(sprintf('Analysis Complete.\nBest Model: %s (%.1f%%)', bestName, accuracies(bestIdx)), 'Success');
    end

    % --- Helper: Train Model ---
    function mdl = trainModel(type, X, Y)
        switch type
            case 'SVM'
                mdl = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', true);
            case 'KNN'
                mdl = fitcknn(X, Y, 'NumNeighbors', 5, 'Standardize', true);
            case 'Ensemble'
                mdl = fitcensemble(X, Y, 'Method', 'Bag', 'NumLearningCycles', 50);
        end
    end

    % --- Helper: Load Data ---
    function [X, Y] = loadAllData(hWait)
        X = [];
        Y = {};
        
        subs = {'sub1', 'sub2', 'sub3', 'sub4'};
        types = {'Normal', 'Aggressive'};
        
        % Pre-calculate filter coefficients ONCE to improve performance
        fs = 4000;
        [b_bp, a_bp] = butter(4, [20 450]/(fs/2), 'bandpass');
        w_notch = [49 51] / (fs/2);
        [b_notch, a_notch] = butter(2, w_notch, 'stop');
        
        totalSteps = length(subs) * length(types);
        step = 0;
        
        for s = 1:length(subs)
            for t = 1:length(types)
                step = step + 1;
                msg = sprintf('Loading %s - %s...', subs{s}, types{t});
                
                % Update waitbar and force draw
                if isvalid(hWait)
                    waitbar(step/totalSteps * 0.5, hWait, msg);
                end
                fprintf('%s\n', msg); % Print progress to command window
                drawnow; 
                
                folderPath = fullfile('EMG Physical Action Data Set', subs{s}, types{t}, 'txt');
                files = dir(fullfile(folderPath, '*.txt'));
                
                for k = 1:length(files)
                    filePath = fullfile(files(k).folder, files(k).name);
                    try
                        raw = load(filePath);
                        
                        if isempty(raw)
                            continue;
                        end
                        
                        filtered = zeros(size(raw));
                        for ch = 1:8
                            % Use pre-calculated coefficients
                            filtered(:, ch) = filtfilt(b_bp, a_bp, raw(:, ch));
                            filtered(:, ch) = filtfilt(b_notch, a_notch, filtered(:, ch));
                        end
                        
                        % Feature Extraction (32 features: 4 types * 8 channels)
                        feats = [];
                        % RMS
                        feats = [feats, rms(filtered)];
                        % MAV
                        feats = [feats, mean(abs(filtered))];
                        % ZC
                        threshold = 0;
                        zc = sum(abs(diff(filtered > threshold))) ./ size(filtered, 1);
                        feats = [feats, zc];
                        % WL
                        wl = sum(abs(diff(filtered)));
                        feats = [feats, wl];
                        
                        X = [X; feats];
                        Y = [Y; types{t}];
                        
                    catch ME
                        fprintf('Error processing file %s: %s\n', files(k).name, ME.message);
                        continue;
                    end
                end
            end
        end
    end
end
