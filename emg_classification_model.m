function emg_classification_model
    % EMG_CLASSIFICATION_MODEL
    % Loads EMG data, extracts features, trains a classification model,
    % and saves the trained model for future use.
    %
    % Features extracted: RMS, MAV, ZC, WL (per channel)
    % Total features: 4 features * 8 channels = 32 features per observation.
    
    % 1. Configuration
    saveModel = true;
    modelFileName = 'trained_emg_model.mat';
    
    % 2. Load and Process Data
    fprintf('--- Starting Data Loading and Feature Extraction ---\n');
    [X, Y] = loadAndExtractFeatures();
    
    if isempty(X)
        errordlg('No data loaded. Check dataset path.', 'Error');
        return;
    end
    
    fprintf('Data loaded: %d samples, %d features.\n', size(X, 1), size(X, 2));
    
    % 3. Train Classifier
    % Using a Bagged Ensemble of Decision Trees (Random Forest equivalent)
    % as it is generally robust and handles non-linearities well.
    fprintf('--- Training Classification Model (Ensemble Bagged Trees) ---\n');
    
    % Split data for evaluation (80% Train, 20% Test)
    cv = cvpartition(Y, 'HoldOut', 0.2);
    XTrain = X(training(cv), :);
    YTrain = Y(training(cv));
    XTest = X(test(cv), :);
    YTest = Y(test(cv));
    
    % Train Model
    tStart = tic;
    trainedModel = fitcensemble(XTrain, YTrain, ...
        'Method', 'Bag', ...
        'NumLearningCycles', 50, ...
        'Learners', templateTree('MaxNumSplits', 20));
    trainTime = toc(tStart);
    
    fprintf('Training completed in %.2f seconds.\n', trainTime);
    
    % 4. Evaluate Model
    fprintf('--- Evaluating Model ---\n');
    YPred = predict(trainedModel, XTest);
    accuracy = sum(strcmp(YPred, YTest)) / length(YTest) * 100;
    
    fprintf('Validation Accuracy: %.2f%%\n', accuracy);
    
    % Display Confusion Matrix
    figure('Name', 'Model Evaluation', 'NumberTitle', 'off', 'Color', 'w');
    confusionchart(YTest, YPred, ...
        'Title', sprintf('Confusion Matrix (Acc: %.1f%%)', accuracy), ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    
    % 5. Save Model
    if saveModel
        fprintf('--- Saving Model ---\n');
        save(modelFileName, 'trainedModel', 'accuracy', 'trainTime');
        fprintf('Model saved to: %s\n', fullfile(pwd, modelFileName));
        msgbox(sprintf('Model Trained & Saved!\nAccuracy: %.1f%%', accuracy), 'Success');
    end

    % ---------------------------------------------------------
    % Helper Function: Load Data & Extract Features
    % ---------------------------------------------------------
    function [X, Y] = loadAndExtractFeatures()
        X = [];
        Y = {};
        
        subs = {'sub1', 'sub2', 'sub3', 'sub4'};
        types = {'Normal', 'Aggressive'};
        
        % Filter Design
        fs = 4000;
        [b_bp, a_bp] = butter(4, [20 450]/(fs/2), 'bandpass');
        w_notch = [49 51] / (fs/2);
        [b_notch, a_notch] = butter(2, w_notch, 'stop');
        
        hWait = waitbar(0, 'Loading Data...');
        totalSteps = length(subs) * length(types);
        step = 0;
        
        for s = 1:length(subs)
            for t = 1:length(types)
                step = step + 1;
                waitbar(step/totalSteps, hWait, sprintf('Loading %s - %s...', subs{s}, types{t}));
                
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
                        % 1. RMS (Root Mean Square)
                        f_rms = rms(filtered);
                        
                        % 2. MAV (Mean Absolute Value)
                        f_mav = mean(abs(filtered));
                        
                        % 3. ZC (Zero Crossing Rate)
                        threshold = 0;
                        f_zc = sum(abs(diff(filtered > threshold))) ./ size(filtered, 1);
                        
                        % 4. WL (Waveform Length)
                        f_wl = sum(abs(diff(filtered)));
                        
                        % Concatenate all features for this sample
                        % [RMS_ch1...RMS_ch8, MAV_ch1...MAV_ch8, etc.]
                        sampleFeatures = [f_rms, f_mav, f_zc, f_wl];
                        
                        X = [X; sampleFeatures];
                        Y = [Y; types{t}];
                        
                    catch
                        continue;
                    end
                end
            end
        end
        delete(hWait);
    end
end
