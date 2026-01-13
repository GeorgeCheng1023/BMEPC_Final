% Compare Parameter Effects on Classification Algorithms
% This script tests different parameter values and compares their effects

clear all
close all
clc

% Load data
fprintf('Loading data...\n');

folder = 'EMG Physical Action Data Set';
subs = {'sub1', 'sub2', 'sub3', 'sub4'};
types = {'Normal', 'Aggressive'};
ch = 8;

X = [];
Y = [];

for i = 1:length(subs)
    for j = 1:length(types)
        path = fullfile(folder, subs{i}, types{j}, 'txt');
        
        if ~isfolder(path)
            continue
        end
        
        files = dir(fullfile(path, '*.txt'));
        
        for k = 1:length(files)
            try
                data = readmatrix(fullfile(path, files(k).name));
                
                if size(data, 2) ~= ch
                    data = data';
                end
                
                feat = extract_features(data);
                feat_row = reshape(feat', 1, []);
                
                X = [X; feat_row];
                Y = [Y; j-1];
            catch
            end
        end
    end
end

fprintf('Loaded %d samples\n\n', size(X,1));

% Normalize
X = zscore(X);

% Setup cross-validation
folds = 5;
rng(100);
cv = cvpartition(Y, 'KFold', folds);

%% Test SVM Linear - BoxConstraint
fprintf('Testing SVM Linear with different BoxConstraint values...\n');

box_vals = [0.1, 0.5, 1, 5, 10, 50, 100];
svm_lin_results = zeros(length(box_vals), 3);

for i = 1:length(box_vals)
    acc_temp = [];
    sens_temp = [];
    spec_temp = [];
    
    for f = 1:folds
        tr = training(cv, f);
        te = test(cv, f);
        
        Xtr = X(tr, :);
        Ytr = Y(tr);
        Xte = X(te, :);
        Yte = Y(te);
        
        [a, s, sp, ~, ~] = train_svm_linear(Xtr, Ytr, Xte, Yte, 'BoxConstraint', box_vals(i));
        
        acc_temp = [acc_temp; a];
        sens_temp = [sens_temp; s];
        spec_temp = [spec_temp; sp];
    end
    
    svm_lin_results(i, 1) = mean(acc_temp) * 100;
    svm_lin_results(i, 2) = mean(sens_temp) * 100;
    svm_lin_results(i, 3) = mean(spec_temp) * 100;
    
    fprintf('  BoxConstraint=%.1f: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%\n', ...
        box_vals(i), svm_lin_results(i, 1), svm_lin_results(i, 2), svm_lin_results(i, 3));
end

%% Test SVM RBF - BoxConstraint and KernelScale
fprintf('\nTesting SVM RBF with different BoxConstraint values...\n');

box_vals_rbf = [0.1, 0.5, 1, 5, 10, 50, 100];
svm_rbf_results = zeros(length(box_vals_rbf), 3);

for i = 1:length(box_vals_rbf)
    acc_temp = [];
    sens_temp = [];
    spec_temp = [];
    
    for f = 1:folds
        tr = training(cv, f);
        te = test(cv, f);
        
        Xtr = X(tr, :);
        Ytr = Y(tr);
        Xte = X(te, :);
        Yte = Y(te);
        
        [a, s, sp, ~, ~] = train_svm_rbf(Xtr, Ytr, Xte, Yte, 'BoxConstraint', box_vals_rbf(i));
        
        acc_temp = [acc_temp; a];
        sens_temp = [sens_temp; s];
        spec_temp = [spec_temp; sp];
    end
    
    svm_rbf_results(i, 1) = mean(acc_temp) * 100;
    svm_rbf_results(i, 2) = mean(sens_temp) * 100;
    svm_rbf_results(i, 3) = mean(spec_temp) * 100;
    
    fprintf('  BoxConstraint=%.1f: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%\n', ...
        box_vals_rbf(i), svm_rbf_results(i, 1), svm_rbf_results(i, 2), svm_rbf_results(i, 3));
end

fprintf('\nTesting SVM RBF with different KernelScale values...\n');

kernel_vals = {0.1, 0.5, 1, 2, 5, 10, 'auto'};
svm_rbf_kernel_results = zeros(length(kernel_vals), 3);

for i = 1:length(kernel_vals)
    acc_temp = [];
    sens_temp = [];
    spec_temp = [];
    
    for f = 1:folds
        tr = training(cv, f);
        te = test(cv, f);
        
        Xtr = X(tr, :);
        Ytr = Y(tr);
        Xte = X(te, :);
        Yte = Y(te);
        
        [a, s, sp, ~, ~] = train_svm_rbf(Xtr, Ytr, Xte, Yte, 'KernelScale', kernel_vals{i});
        
        acc_temp = [acc_temp; a];
        sens_temp = [sens_temp; s];
        spec_temp = [spec_temp; sp];
    end
    
    svm_rbf_kernel_results(i, 1) = mean(acc_temp) * 100;
    svm_rbf_kernel_results(i, 2) = mean(sens_temp) * 100;
    svm_rbf_kernel_results(i, 3) = mean(spec_temp) * 100;
    
    if isnumeric(kernel_vals{i})
        fprintf('  KernelScale=%.1f: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%\n', ...
            kernel_vals{i}, svm_rbf_kernel_results(i, 1), svm_rbf_kernel_results(i, 2), svm_rbf_kernel_results(i, 3));
    else
        fprintf('  KernelScale=%s: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%\n', ...
            kernel_vals{i}, svm_rbf_kernel_results(i, 1), svm_rbf_kernel_results(i, 2), svm_rbf_kernel_results(i, 3));
    end
end

%% Test k-NN - NumNeighbors
fprintf('\nTesting k-NN with different k values...\n');

k_vals = [1, 3, 5, 7, 9, 11, 15, 20];
knn_results = zeros(length(k_vals), 3);

for i = 1:length(k_vals)
    acc_temp = [];
    sens_temp = [];
    spec_temp = [];
    
    for f = 1:folds
        tr = training(cv, f);
        te = test(cv, f);
        
        Xtr = X(tr, :);
        Ytr = Y(tr);
        Xte = X(te, :);
        Yte = Y(te);
        
        [a, s, sp, ~, ~] = train_knn(Xtr, Ytr, Xte, Yte, 'NumNeighbors', k_vals(i));
        
        acc_temp = [acc_temp; a];
        sens_temp = [sens_temp; s];
        spec_temp = [spec_temp; sp];
    end
    
    knn_results(i, 1) = mean(acc_temp) * 100;
    knn_results(i, 2) = mean(sens_temp) * 100;
    knn_results(i, 3) = mean(spec_temp) * 100;
    
    fprintf('  k=%d: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%\n', ...
        k_vals(i), knn_results(i, 1), knn_results(i, 2), knn_results(i, 3));
end

%% Test Decision Tree - MaxNumSplits
fprintf('\nTesting Decision Tree with different MaxNumSplits values...\n');

split_vals = [5, 10, 20, 30, 40, 50, 100];
tree_results = zeros(length(split_vals), 3);

for i = 1:length(split_vals)
    acc_temp = [];
    sens_temp = [];
    spec_temp = [];
    
    for f = 1:folds
        tr = training(cv, f);
        te = test(cv, f);
        
        Xtr = X(tr, :);
        Ytr = Y(tr);
        Xte = X(te, :);
        Yte = Y(te);
        
        [a, s, sp, ~, ~] = train_decision_tree(Xtr, Ytr, Xte, Yte, 'MaxNumSplits', split_vals(i));
        
        acc_temp = [acc_temp; a];
        sens_temp = [sens_temp; s];
        spec_temp = [spec_temp; sp];
    end
    
    tree_results(i, 1) = mean(acc_temp) * 100;
    tree_results(i, 2) = mean(sens_temp) * 100;
    tree_results(i, 3) = mean(spec_temp) * 100;
    
    fprintf('  MaxNumSplits=%d: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%\n', ...
        split_vals(i), tree_results(i, 1), tree_results(i, 2), tree_results(i, 3));
end

fprintf('\nTesting Decision Tree with different MinLeafSize values...\n');

leaf_vals = [1, 5, 10, 15, 20, 25, 30];
tree_leaf_results = zeros(length(leaf_vals), 3);

for i = 1:length(leaf_vals)
    acc_temp = [];
    sens_temp = [];
    spec_temp = [];
    
    for f = 1:folds
        tr = training(cv, f);
        te = test(cv, f);
        
        Xtr = X(tr, :);
        Ytr = Y(tr);
        Xte = X(te, :);
        Yte = Y(te);
        
        [a, s, sp, ~, ~] = train_decision_tree(Xtr, Ytr, Xte, Yte, 'MinLeafSize', leaf_vals(i));
        
        acc_temp = [acc_temp; a];
        sens_temp = [sens_temp; s];
        spec_temp = [spec_temp; sp];
    end
    
    tree_leaf_results(i, 1) = mean(acc_temp) * 100;
    tree_leaf_results(i, 2) = mean(sens_temp) * 100;
    tree_leaf_results(i, 3) = mean(spec_temp) * 100;
    
    fprintf('  MinLeafSize=%d: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%\n', ...
        leaf_vals(i), tree_leaf_results(i, 1), tree_leaf_results(i, 2), tree_leaf_results(i, 3));
end

%% Create Visualization
fprintf('\nCreating comparison plots...\n');

fig = figure('Name', 'Parameter Comparison', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 900]);

% SVM Linear BoxConstraint
subplot(3, 3, 1);
plot(box_vals, svm_lin_results(:, 1), 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('BoxConstraint');
ylabel('Accuracy (%)');
title('SVM Linear - BoxConstraint');
grid on;
set(gca, 'XScale', 'log');

subplot(3, 3, 2);
plot(box_vals, svm_lin_results(:, 2), 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(box_vals, svm_lin_results(:, 3), 'g-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('BoxConstraint');
ylabel('Rate (%)');
title('SVM Linear - Sensitivity & Specificity');
legend('Sensitivity', 'Specificity', 'Location', 'best');
grid on;
set(gca, 'XScale', 'log');

% SVM RBF BoxConstraint
subplot(3, 3, 3);
plot(box_vals_rbf, svm_rbf_results(:, 1), 'b-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('BoxConstraint');
ylabel('Accuracy (%)');
title('SVM RBF - BoxConstraint');
grid on;
set(gca, 'XScale', 'log');

% SVM RBF KernelScale
subplot(3, 3, 4);
numeric_kernel = [0.1, 0.5, 1, 2, 5, 10];
plot(numeric_kernel, svm_rbf_kernel_results(1:6, 1), 'b-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('KernelScale');
ylabel('Accuracy (%)');
title('SVM RBF - KernelScale');
grid on;
set(gca, 'XScale', 'log');

% k-NN
subplot(3, 3, 5);
plot(k_vals, knn_results(:, 1), 'b-d', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Neighbors (k)');
ylabel('Accuracy (%)');
title('k-NN - NumNeighbors');
grid on;

subplot(3, 3, 6);
plot(k_vals, knn_results(:, 2), 'r-d', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(k_vals, knn_results(:, 3), 'g-d', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Neighbors (k)');
ylabel('Rate (%)');
title('k-NN - Sensitivity & Specificity');
legend('Sensitivity', 'Specificity', 'Location', 'best');
grid on;

% Decision Tree MaxNumSplits
subplot(3, 3, 7);
plot(split_vals, tree_results(:, 1), 'b-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('MaxNumSplits');
ylabel('Accuracy (%)');
title('Decision Tree - MaxNumSplits');
grid on;

% Decision Tree MinLeafSize
subplot(3, 3, 8);
plot(leaf_vals, tree_leaf_results(:, 1), 'b-v', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('MinLeafSize');
ylabel('Accuracy (%)');
title('Decision Tree - MinLeafSize');
grid on;

% Summary comparison
subplot(3, 3, 9);
[~, best_svm_lin] = max(svm_lin_results(:, 1));
[~, best_svm_rbf] = max(svm_rbf_results(:, 1));
[~, best_knn] = max(knn_results(:, 1));
[~, best_tree_split] = max(tree_results(:, 1));

best_accs = [svm_lin_results(best_svm_lin, 1), ...
             svm_rbf_results(best_svm_rbf, 1), ...
             knn_results(best_knn, 1), ...
             tree_results(best_tree_split, 1)];

bar(best_accs);
set(gca, 'XTickLabel', {'SVM Lin', 'SVM RBF', 'k-NN', 'Tree'});
ylabel('Best Accuracy (%)');
title('Best Performance Comparison');
grid on;
ylim([min(best_accs)-5, max(best_accs)+5]);

%% Print Best Results
fprintf('\n=== BEST PARAMETERS ===\n\n');

[best_acc_lin, idx_lin] = max(svm_lin_results(:, 1));
fprintf('SVM Linear:\n');
fprintf('  Best BoxConstraint = %.1f\n', box_vals(idx_lin));
fprintf('  Accuracy = %.2f%%\n', best_acc_lin);
fprintf('  Sensitivity = %.2f%%\n', svm_lin_results(idx_lin, 2));
fprintf('  Specificity = %.2f%%\n\n', svm_lin_results(idx_lin, 3));

[best_acc_rbf, idx_rbf] = max(svm_rbf_results(:, 1));
fprintf('SVM RBF (BoxConstraint):\n');
fprintf('  Best BoxConstraint = %.1f\n', box_vals_rbf(idx_rbf));
fprintf('  Accuracy = %.2f%%\n', best_acc_rbf);
fprintf('  Sensitivity = %.2f%%\n', svm_rbf_results(idx_rbf, 2));
fprintf('  Specificity = %.2f%%\n\n', svm_rbf_results(idx_rbf, 3));

[best_acc_kern, idx_kern] = max(svm_rbf_kernel_results(:, 1));
fprintf('SVM RBF (KernelScale):\n');
if isnumeric(kernel_vals(idx_kern))
    fprintf('  Best KernelScale = %.1f\n', kernel_vals(idx_kern));
else
    fprintf('  Best KernelScale = %s\n', kernel_vals{idx_kern});
end
fprintf('  Accuracy = %.2f%%\n', best_acc_kern);
fprintf('  Sensitivity = %.2f%%\n', svm_rbf_kernel_results(idx_kern, 2));
fprintf('  Specificity = %.2f%%\n\n', svm_rbf_kernel_results(idx_kern, 3));

[best_acc_knn, idx_knn] = max(knn_results(:, 1));
fprintf('k-NN:\n');
fprintf('  Best k = %d\n', k_vals(idx_knn));
fprintf('  Accuracy = %.2f%%\n', best_acc_knn);
fprintf('  Sensitivity = %.2f%%\n', knn_results(idx_knn, 2));
fprintf('  Specificity = %.2f%%\n\n', knn_results(idx_knn, 3));

[best_acc_tree, idx_tree] = max(tree_results(:, 1));
fprintf('Decision Tree (MaxNumSplits):\n');
fprintf('  Best MaxNumSplits = %d\n', split_vals(idx_tree));
fprintf('  Accuracy = %.2f%%\n', best_acc_tree);
fprintf('  Sensitivity = %.2f%%\n', tree_results(idx_tree, 2));
fprintf('  Specificity = %.2f%%\n\n', tree_results(idx_tree, 3));

[best_acc_leaf, idx_leaf] = max(tree_leaf_results(:, 1));
fprintf('Decision Tree (MinLeafSize):\n');
fprintf('  Best MinLeafSize = %d\n', leaf_vals(idx_leaf));
fprintf('  Accuracy = %.2f%%\n', best_acc_leaf);
fprintf('  Sensitivity = %.2f%%\n', tree_leaf_results(idx_leaf, 2));
fprintf('  Specificity = %.2f%%\n\n', tree_leaf_results(idx_leaf, 3));

fprintf('Done!\n');

% Save results
save('parameter_comparison_results.mat', 'svm_lin_results', 'svm_rbf_results', ...
     'svm_rbf_kernel_results', 'knn_results', 'tree_results', 'tree_leaf_results', ...
     'box_vals', 'box_vals_rbf', 'kernel_vals', 'k_vals', 'split_vals', 'leaf_vals');

% Helper function
function feat = extract_features(data)
    n = size(data, 2);
    feat = zeros(3, n);
    
    for i = 1:n
        sig = data(:, i);
        feat(1, i) = rms(sig);
        feat(2, i) = sum(abs(diff(sig)));
        feat(3, i) = var(sig);
    end
end
