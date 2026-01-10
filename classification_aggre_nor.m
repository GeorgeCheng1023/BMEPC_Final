% EMG Classification - Normal vs Aggressive

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

% Train models
folds = 5;
cv = cvpartition(Y, 'KFold', folds);

fprintf('Training models...\n');

% SVM Linear
svm_lin_acc = [];
svm_lin_sens = [];
svm_lin_spec = [];
svm_lin_scores = [];
svm_lin_labels = [];

% SVM RBF
svm_rbf_acc = [];
svm_rbf_sens = [];
svm_rbf_spec = [];
svm_rbf_scores = [];
svm_rbf_labels = [];

% k-NN
knn_acc = [];
knn_sens = [];
knn_spec = [];
knn_scores = [];
knn_labels = [];

% Decision Tree
tree_acc = [];
tree_sens = [];
tree_spec = [];
tree_scores = [];
tree_labels = [];

for f = 1:folds
    tr = training(cv, f);
    te = test(cv, f);
    
    Xtr = X(tr, :);
    Ytr = Y(tr);
    Xte = X(te, :);
    Yte = Y(te);
    
    % SVM Linear
    m1 = fitcsvm(Xtr, Ytr, 'KernelFunction', 'linear', 'Standardize', false);
    [p1, sc1] = predict(m1, Xte);
    [a1, s1, sp1] = calculate_metrics(Yte, p1);
    svm_lin_acc = [svm_lin_acc; a1];
    svm_lin_sens = [svm_lin_sens; s1];
    svm_lin_spec = [svm_lin_spec; sp1];
    svm_lin_scores = [svm_lin_scores; sc1(:, 2)];
    svm_lin_labels = [svm_lin_labels; Yte];
    
    % SVM RBF
    m2 = fitcsvm(Xtr, Ytr, 'KernelFunction', 'rbf', 'Standardize', false);
    [p2, sc2] = predict(m2, Xte);
    [a2, s2, sp2] = calculate_metrics(Yte, p2);
    svm_rbf_acc = [svm_rbf_acc; a2];
    svm_rbf_sens = [svm_rbf_sens; s2];
    svm_rbf_spec = [svm_rbf_spec; sp2];
    svm_rbf_scores = [svm_rbf_scores; sc2(:, 2)];
    svm_rbf_labels = [svm_rbf_labels; Yte];
    
    % k-NN
    m3 = fitcknn(Xtr, Ytr, 'NumNeighbors', 5, 'Distance', 'euclidean');
    [p3, sc3] = predict(m3, Xte);
    [a3, s3, sp3] = calculate_metrics(Yte, p3);
    knn_acc = [knn_acc; a3];
    knn_sens = [knn_sens; s3];
    knn_spec = [knn_spec; sp3];
    knn_scores = [knn_scores; sc3(:, 2)];
    knn_labels = [knn_labels; Yte];
    
    % Decision Tree
    m4 = fitctree(Xtr, Ytr);
    [p4, sc4] = predict(m4, Xte);
    [a4, s4, sp4] = calculate_metrics(Yte, p4);
    tree_acc = [tree_acc; a4];
    tree_sens = [tree_sens; s4];
    tree_spec = [tree_spec; sp4];
    tree_scores = [tree_scores; sc4(:, 2)];
    tree_labels = [tree_labels; Yte];
end

% Results
fprintf('\n=== RESULTS ===\n\n');

names = {'SVM (Linear)'; 'SVM (RBF)'; 'k-NN (k=5)'; 'Decision Tree'};
acc = [mean(svm_lin_acc)*100; mean(svm_rbf_acc)*100; mean(knn_acc)*100; mean(tree_acc)*100];
sens = [mean(svm_lin_sens)*100; mean(svm_rbf_sens)*100; mean(knn_sens)*100; mean(tree_sens)*100];
spec = [mean(svm_lin_spec)*100; mean(svm_rbf_spec)*100; mean(knn_spec)*100; mean(tree_spec)*100];

tbl = table(names, acc, sens, spec, 'VariableNames', {'Model', 'Accuracy', 'Sensitivity', 'Specificity'});
disp(tbl);

[best_val, best_idx] = max(acc);
fprintf('\nBest: %s (%.2f%%)\n\n', names{best_idx}, best_val);

% Plots
fprintf('Creating plots...\n');

fig = figure('Name', 'EMG Results', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 700]);
tg = uitabgroup(fig);

% Tab 1: Confusion Matrices
tab1 = uitab(tg, 'Title', 'Confusion Matrices');

m1_all = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', false);
p1_all = predict(m1_all, X);

m2_all = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'Standardize', false);
p2_all = predict(m2_all, X);

m3_all = fitcknn(X, Y, 'NumNeighbors', 5, 'Distance', 'euclidean');
p3_all = predict(m3_all, X);

m4_all = fitctree(X, Y);
p4_all = predict(m4_all, X);

subplot(2, 2, 1, 'Parent', tab1);
confusionchart(Y, p1_all, 'Title', 'SVM (Linear)', 'Normalization', 'row-normalized');

subplot(2, 2, 2, 'Parent', tab1);
confusionchart(Y, p2_all, 'Title', 'SVM (RBF)', 'Normalization', 'row-normalized');

subplot(2, 2, 3, 'Parent', tab1);
confusionchart(Y, p3_all, 'Title', 'k-NN (k=5)', 'Normalization', 'row-normalized');

subplot(2, 2, 4, 'Parent', tab1);
confusionchart(Y, p4_all, 'Title', 'Decision Tree', 'Normalization', 'row-normalized');

% Tab 2: ROC Curves
tab2 = uitab(tg, 'Title', 'ROC Curves');
ax_roc = axes('Parent', tab2);
hold(ax_roc, 'on');

[fpr1, tpr1, ~, auc1] = perfcurve(svm_lin_labels, svm_lin_scores, 1);
[fpr2, tpr2, ~, auc2] = perfcurve(svm_rbf_labels, svm_rbf_scores, 1);
[fpr3, tpr3, ~, auc3] = perfcurve(knn_labels, knn_scores, 1);
[fpr4, tpr4, ~, auc4] = perfcurve(tree_labels, tree_scores, 1);

plot(ax_roc, fpr1, tpr1, 'LineWidth', 2.5, 'DisplayName', sprintf('SVM Linear (AUC=%.3f)', auc1));
plot(ax_roc, fpr2, tpr2, 'LineWidth', 2.5, 'DisplayName', sprintf('SVM RBF (AUC=%.3f)', auc2));
plot(ax_roc, fpr3, tpr3, 'LineWidth', 2.5, 'DisplayName', sprintf('k-NN (AUC=%.3f)', auc3));
plot(ax_roc, fpr4, tpr4, 'LineWidth', 2.5, 'DisplayName', sprintf('Decision Tree (AUC=%.3f)', auc4));
plot(ax_roc, [0 1], [0 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Random');

xlabel(ax_roc, 'False Positive Rate', 'FontSize', 12);
ylabel(ax_roc, 'True Positive Rate', 'FontSize', 12);
title(ax_roc, 'ROC Curves', 'FontSize', 14);
legend(ax_roc, 'Location', 'southeast', 'FontSize', 11);
grid(ax_roc, 'on');
axis(ax_roc, 'square');
hold(ax_roc, 'off');

% Tab 3: Decision Boundary
tab3 = uitab(tg, 'Title', 'Decision Boundary');
ax_db = axes('Parent', tab3);
hold(ax_db, 'on');

m_lin = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', false);

Xviz = X(:, 1:2);

xmin = min(Xviz(:, 1)) - 0.5;
xmax = max(Xviz(:, 1)) + 0.5;
ymin = min(Xviz(:, 2)) - 0.5;
ymax = max(Xviz(:, 2)) + 0.5;

[xx, yy] = meshgrid(linspace(xmin, xmax, 100), linspace(ymin, ymax, 100));

Xmesh = [xx(:), yy(:), repmat(mean(X(:, 3:end)), size(xx(:), 1), 1)];

Z = predict(m_lin, Xmesh);
Z = reshape(Z, size(xx));

contourf(ax_db, xx, yy, Z, 2, 'LineWidth', 1.5, 'LineColor', 'k');
colormap(ax_db, [0.9 0.95 1.0; 1.0 0.9 0.9]);

idx0 = Y == 0;
idx1 = Y == 1;

scatter(ax_db, Xviz(idx0, 1), Xviz(idx0, 2), 80, 'b', 'o', 'filled', 'DisplayName', 'Normal');
scatter(ax_db, Xviz(idx1, 1), Xviz(idx1, 2), 80, 'r', '^', 'filled', 'DisplayName', 'Aggressive');

xlabel(ax_db, 'Feature 1 (RMS - Ch1)', 'FontSize', 12);
ylabel(ax_db, 'Feature 2 (WL - Ch1)', 'FontSize', 12);
title(ax_db, 'SVM Linear Decision Boundary', 'FontSize', 14);
legend(ax_db, 'Location', 'best', 'FontSize', 11);
grid(ax_db, 'on');
ax_db.GridAlpha = 0.3;
hold(ax_db, 'off');

fprintf('Done!\n');

save('classification_aggre_nor_results.mat', 'acc', 'sens', 'spec');

% Helper functions

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

function [a, s, sp] = calculate_metrics(true_y, pred_y)
    tp = sum((true_y == 1) & (pred_y == 1));
    tn = sum((true_y == 0) & (pred_y == 0));
    fp = sum((true_y == 0) & (pred_y == 1));
    fn = sum((true_y == 1) & (pred_y == 0));
    
    a = (tp + tn) / (tp + tn + fp + fn);
    s = tp / (tp + fn);
    sp = tn / (tn + fp);
end
