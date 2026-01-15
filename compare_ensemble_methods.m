clear all
close all
clc

fprintf('Loading data...\n');

folder = 'EMG Physical Action Data Set - del';
subs = {'sub1', 'sub2', 'sub3', 'sub4'};

normalActions = {'Clapping', 'Handshaking', 'Hugging', 'Waving'};
aggressiveActions = {'Elbowing', 'Hamering', 'Pulling', 'Punching', 'Pushing', 'Slapping'};

ch = 8;
targetChannels = 1:8;

X = [];
Y_class = [];
Y_normal_action = [];
Y_aggr_action = [];

for i = 1:length(subs)
    
    for j = 1:length(normalActions)
        path = fullfile(folder, subs{i}, 'Normal', 'txt', [normalActions{j}, '.txt']);
        
        if ~isfile(path)
            continue
        end
        
        try
            data = readmatrix(path);
            
            if size(data, 2) ~= ch
                data = data';
            end
            
            data = data(:, targetChannels);
            
            windowSize = 256;
            stepSize = 128;
            numSamples = size(data, 1);
            
            for w = 1:stepSize:(numSamples - windowSize + 1)
                windowData = data(w:w+windowSize-1, :);
                
                feat = extract_features(windowData);
                feat_row = reshape(feat', 1, []);
                
                X = [X; feat_row];
                Y_class = [Y_class; 0];
                Y_normal_action = [Y_normal_action; j-1];
                Y_aggr_action = [Y_aggr_action; -1];
            end
        catch
        end
    end
    
    for j = 1:length(aggressiveActions)
        path = fullfile(folder, subs{i}, 'Aggressive', 'txt', [aggressiveActions{j}, '.txt']);
        
        if ~isfile(path)
            continue
        end
        
        try
            data = readmatrix(path);
            
            if size(data, 2) ~= ch
                data = data';
            end
            
            data = data(:, targetChannels);
            
            windowSize = 256;
            stepSize = 128;
            numSamples = size(data, 1);
            
            for w = 1:stepSize:(numSamples - windowSize + 1)
                windowData = data(w:w+windowSize-1, :);
                
                feat = extract_features(windowData);
                feat_row = reshape(feat', 1, []);
                
                X = [X; feat_row];
                Y_class = [Y_class; 1];
                Y_normal_action = [Y_normal_action; -1];
                Y_aggr_action = [Y_aggr_action; j-1];
            end
        catch
        end
    end
end

fprintf('Loaded %d samples\n\n', size(X,1));

X = zscore(X);

idx_aggr = Y_class == 1;
X_aggr = X(idx_aggr, :);
Y_aggr = Y_aggr_action(idx_aggr);

fprintf('Testing aggressive action classification with different methods...\n\n');

folds = 5;
rng(100);
cv = cvpartition(Y_aggr, 'KFold', folds);

methods = {'TotalBoost', 'RUSBoost', 'LPBoost', 'Subspace', 'Bag'};
accuracies = zeros(length(methods), 1);
times = zeros(length(methods), 1);

for m = 1:length(methods)
    fprintf('Testing %s...\n', methods{m});
    
    tic;
    method_acc = [];
    
    for f = 1:folds
        tr = training(cv, f);
        te = test(cv, f);
        
        Xtr = X_aggr(tr, :);
        Ytr = Y_aggr(tr);
        Xte = X_aggr(te, :);
        Yte = Y_aggr(te);
        
        try
            if strcmp(methods{m}, 'TotalBoost')
                template = templateTree('MaxNumSplits', 50, 'MinLeafSize', 2);
                mdl = fitcensemble(Xtr, Ytr, 'Method', 'TotalBoost', 'NumLearningCycles', 200, 'Learners', template);
            elseif strcmp(methods{m}, 'RUSBoost')
                mdl = fitcensemble(Xtr, Ytr, 'Method', 'RUSBoost', 'NumLearningCycles', 200, 'Learners', 'tree', 'LearnRate', 0.1);
            elseif strcmp(methods{m}, 'LPBoost')
                template = templateTree('MaxNumSplits', 50, 'MinLeafSize', 2);
                mdl = fitcensemble(Xtr, Ytr, 'Method', 'LPBoost', 'NumLearningCycles', 200, 'Learners', template);
            elseif strcmp(methods{m}, 'Subspace')
                template = templateTree('MaxNumSplits', 50, 'MinLeafSize', 2);
                mdl = fitcensemble(Xtr, Ytr, 'Method', 'Subspace', 'NumLearningCycles', 200, 'Learners', template);
            elseif strcmp(methods{m}, 'Bag')
                template = templateTree('MaxNumSplits', 50, 'MinLeafSize', 2, 'NumVariablesToSample', 'all');
                mdl = fitcensemble(Xtr, Ytr, 'Method', 'Bag', 'NumLearningCycles', 200, 'Learners', template);
            end
            
            pred = predict(mdl, Xte);
            acc = sum(pred == Yte) / length(Yte);
            method_acc = [method_acc; acc];
        catch err
            fprintf('  Error: %s\n', err.message);
            method_acc = [method_acc; 0];
        end
    end
    
    elapsed = toc;
    
    accuracies(m) = mean(method_acc) * 100;
    times(m) = elapsed;
    
    fprintf('  Accuracy: %.2f%%\n', accuracies(m));
    fprintf('  Time: %.2f seconds\n\n', times(m));
end

fprintf('\n=== COMPARISON RESULTS ===\n\n');

tbl = table(methods', accuracies, times, 'VariableNames', {'Method', 'Accuracy', 'Time_sec'});
disp(tbl);

[best_acc, best_idx] = max(accuracies);
fprintf('\nBest Method: %s (%.2f%%)\n', methods{best_idx}, best_acc);

fig = figure('Name', 'Ensemble Method Comparison', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 500]);

subplot(1, 2, 1);
bar(accuracies, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 15);
ylabel('Accuracy (%)', 'FontSize', 12);
title('Accuracy Comparison', 'FontSize', 14);
ylim([0 100]);
grid on;

subplot(1, 2, 2);
bar(times, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 15);
ylabel('Time (seconds)', 'FontSize', 12);
title('Training Time Comparison', 'FontSize', 14);
grid on;

fprintf('\nDone!\n');

function feat = extract_features(data)
    n = size(data, 2);
    feat = zeros(24, n);
    
    for i = 1:n
        sig = data(:, i);
        N = length(sig);
        
        feat(1, i) = rms(sig);
        feat(2, i) = sum(abs(diff(sig)));
        feat(3, i) = var(sig);
        feat(4, i) = mean(abs(sig));
        feat(5, i) = max(abs(sig));
        feat(6, i) = median(abs(sig));
        
        threshold = mean(abs(sig));
        zc = sum(abs(diff(sig > threshold)));
        feat(7, i) = zc;
        
        ssc = 0;
        for j = 2:length(sig)-1
            if (sig(j) - sig(j-1)) * (sig(j) - sig(j+1)) > 0
                ssc = ssc + 1;
            end
        end
        feat(8, i) = ssc;
        
        Y = fft(sig);
        P2 = abs(Y/N);
        P1 = P2(1:N/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        
        feat(9, i) = mean(P1);
        feat(10, i) = median(P1);
        feat(11, i) = max(P1);
        
        wl_ratio = sum(abs(diff(sig))) / length(sig);
        feat(12, i) = wl_ratio;
        
        feat(13, i) = std(sig);
        feat(14, i) = skewness(sig);
        feat(15, i) = kurtosis(sig);
        
        [~, maxIdx] = max(P1);
        feat(16, i) = maxIdx;
        
        feat(17, i) = sum(abs(sig));
        
        wamp_threshold = 0.05;
        wamp = 0;
        for j = 1:length(sig)-1
            if abs(sig(j+1) - sig(j)) > wamp_threshold
                wamp = wamp + 1;
            end
        end
        feat(18, i) = wamp;
        
        feat(19, i) = log(sum(abs(sig)));
        
        feat(20, i) = sum(sig.^2);
        
        freqs = (0:N/2) / N;
        P1_norm = P1 / sum(P1);
        feat(21, i) = sum(freqs' .* P1_norm);
        
        feat(22, i) = sqrt(sum(((freqs' - feat(21, i)).^2) .* P1_norm));
        
        ac = xcorr(sig, 'coeff');
        feat(23, i) = ac(N+1);
        
        energy_ratio = sum(sig(1:floor(N/2)).^2) / sum(sig.^2);
        feat(24, i) = energy_ratio;
    end
end
