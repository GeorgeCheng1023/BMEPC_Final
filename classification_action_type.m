% EMG Action-Type Classification (multi-class)

clear all
close all
clc

fprintf('Loading data for action-type classification...\n');

folder = 'EMG Physical Action Data Set - del';
subs = {'sub1', 'sub2', 'sub3', 'sub4'};
types = {'Normal','Aggressive'};
ch = 8;

normal_actions = {'Bowing','Clapping','Handshaking','Hugging','Jumping','Running','Seating','Standing','Walking','Waving'};
aggressive_actions = {'Elbowing','Frontkicking','Hamering','Headering','Kneeing','Pulling','Punching','Pushing','Sidekicking','Slapping'};
class_names = [normal_actions aggressive_actions];
class_ids = 1:numel(class_names);
class_map = containers.Map(class_names, class_ids);

X = [];
Y = [];
labels_read = [];

for i = 1:numel(subs)
    sub_name = subs{i};
    for t = 1:numel(types)
        type_name = types{t};
        path = fullfile(folder, sub_name, type_name, 'txt');

        if ~isfolder(path)
            continue
        end

        files = dir(fullfile(path, '*.txt'));

        for k = 1:numel(files)
            file_path = fullfile(path, files(k).name);
            [~, base, ~] = fileparts(files(k).name);

            if ~isKey(class_map, base)
                continue
            end

            try
                data = readmatrix(file_path);
                if size(data, 2) ~= ch
                    data = data';
                end

                feat = extract_features(data);
                feat_row = reshape(feat', 1, []);

                X = [X; feat_row]; %#ok<AGROW>
                Y = [Y; class_map(base)]; %#ok<AGROW>
                labels_read = [labels_read; {base}]; %#ok<AGROW>
            catch
            end
        end
    end
end

classes_in_use = unique(Y)';
class_names_in_use = class_names(classes_in_use);

fprintf('Loaded %d samples across %d classes.\n', size(X, 1), numel(classes_in_use));

if isempty(X)
    error('No samples were loaded. Check dataset paths.');
end

X = zscore(X);

folds = 5;
cv = cvpartition(Y, 'KFold', folds);

fprintf('Training multi-class models with %d-fold CV...\n', folds);

% Metrics holders
svm_lin_acc = [];
svm_lin_sens = [];
svm_lin_spec = [];
svm_lin_scores = [];
svm_lin_labels = [];

svm_rbf_acc = [];
svm_rbf_sens = [];
svm_rbf_spec = [];
svm_rbf_scores = [];
svm_rbf_labels = [];

knn_acc = [];
knn_sens = [];
knn_spec = [];
knn_scores = [];
knn_labels = [];

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

    % SVM Linear ECOC
    t_lin = templateSVM('KernelFunction', 'linear', 'Standardize', false);
    m1 = fitcecoc(Xtr, Ytr, 'Learners', t_lin, 'ClassNames', classes_in_use);
    [p1, sc1] = predict(m1, Xte);
    [a1, s1, sp1] = multiclass_metrics(Yte, p1, classes_in_use);
    svm_lin_acc = [svm_lin_acc; a1]; %#ok<AGROW>
    svm_lin_sens = [svm_lin_sens; s1]; %#ok<AGROW>
    svm_lin_spec = [svm_lin_spec; sp1]; %#ok<AGROW>
    svm_lin_scores = [svm_lin_scores; sc1]; %#ok<AGROW>
    svm_lin_labels = [svm_lin_labels; Yte]; %#ok<AGROW>

    % SVM RBF ECOC
    t_rbf = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
    m2 = fitcecoc(Xtr, Ytr, 'Learners', t_rbf, 'ClassNames', classes_in_use);
    [p2, sc2] = predict(m2, Xte);
    [a2, s2, sp2] = multiclass_metrics(Yte, p2, classes_in_use);
    svm_rbf_acc = [svm_rbf_acc; a2]; %#ok<AGROW>
    svm_rbf_sens = [svm_rbf_sens; s2]; %#ok<AGROW>
    svm_rbf_spec = [svm_rbf_spec; sp2]; %#ok<AGROW>
    svm_rbf_scores = [svm_rbf_scores; sc2]; %#ok<AGROW>
    svm_rbf_labels = [svm_rbf_labels; Yte]; %#ok<AGROW>

    % k-NN (k=5)
    m3 = fitcknn(Xtr, Ytr, 'NumNeighbors', 5, 'Distance', 'euclidean', 'ClassNames', classes_in_use);
    [p3, sc3] = predict(m3, Xte);
    [a3, s3, sp3] = multiclass_metrics(Yte, p3, classes_in_use);
    knn_acc = [knn_acc; a3]; %#ok<AGROW>
    knn_sens = [knn_sens; s3]; %#ok<AGROW>
    knn_spec = [knn_spec; sp3]; %#ok<AGROW>
    knn_scores = [knn_scores; sc3]; %#ok<AGROW>
    knn_labels = [knn_labels; Yte]; %#ok<AGROW>

    % Decision Tree
    m4 = fitctree(Xtr, Ytr, 'ClassNames', classes_in_use);
    [p4, sc4] = predict(m4, Xte);
    [a4, s4, sp4] = multiclass_metrics(Yte, p4, classes_in_use);
    tree_acc = [tree_acc; a4]; %#ok<AGROW>
    tree_sens = [tree_sens; s4]; %#ok<AGROW>
    tree_spec = [tree_spec; sp4]; %#ok<AGROW>
    tree_scores = [tree_scores; sc4]; %#ok<AGROW>
    tree_labels = [tree_labels; Yte]; %#ok<AGROW>
end

% Aggregate metrics
names = {'SVM (Linear)'; 'SVM (RBF)'; 'k-NN (k=5)'; 'Decision Tree'};
acc = [mean(svm_lin_acc); mean(svm_rbf_acc); mean(knn_acc); mean(tree_acc)] * 100;
sens = [mean(svm_lin_sens); mean(svm_rbf_sens); mean(knn_sens); mean(tree_sens)] * 100;
spec = [mean(svm_lin_spec); mean(svm_rbf_spec); mean(knn_spec); mean(tree_spec)] * 100;

% Micro-average ROC/AUC
[roc1_fpr, roc1_tpr, roc1_auc] = micro_roc(svm_lin_labels, svm_lin_scores, classes_in_use);
[roc2_fpr, roc2_tpr, roc2_auc] = micro_roc(svm_rbf_labels, svm_rbf_scores, classes_in_use);
[roc3_fpr, roc3_tpr, roc3_auc] = micro_roc(knn_labels, knn_scores, classes_in_use);
[roc4_fpr, roc4_tpr, roc4_auc] = micro_roc(tree_labels, tree_scores, classes_in_use);

auc = [roc1_auc; roc2_auc; roc3_auc; roc4_auc];

results = table(names, acc, sens, spec, auc, ...
    'VariableNames', {'Model', 'Accuracy', 'MacroSensitivity', 'MacroSpecificity', 'MicroAUC'});
disp('=== MULTI-CLASS RESULTS ===');
disp(results);

[best_val, best_idx] = max(acc);
fprintf('\nBest model: %s (Accuracy %.2f%%)\n', names{best_idx}, best_val);

% Fit best model on full data for confusion matrix
switch best_idx
    case 1
        best_model = fitcecoc(X, Y, 'Learners', templateSVM('KernelFunction', 'linear', 'Standardize', false), ...
            'ClassNames', classes_in_use);
    case 2
        best_model = fitcecoc(X, Y, 'Learners', templateSVM('KernelFunction', 'rbf', 'Standardize', false), ...
            'ClassNames', classes_in_use);
    case 3
        best_model = fitcknn(X, Y, 'NumNeighbors', 5, 'Distance', 'euclidean', 'ClassNames', classes_in_use);
    otherwise
        best_model = fitctree(X, Y, 'ClassNames', classes_in_use);
end

p_best = predict(best_model, X);

% Plots
fig = figure('Name', 'EMG Action-Type Classification', 'NumberTitle', 'off', 'Position', [80, 80, 1200, 700]);
tg = uitabgroup(fig);

% Confusion matrix tab
conf_tab = uitab(tg, 'Title', 'Confusion');
confusionchart(Y, p_best, 'Parent', conf_tab, 'Title', sprintf('Best: %s', names{best_idx}), ...
    'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized', 'DiagonalColor', [0.25 0.6 0.2]);

% ROC tab (micro-average)
roc_tab = uitab(tg, 'Title', 'ROC (Micro)');
ax_roc = axes('Parent', roc_tab);
hold(ax_roc, 'on');
plot(ax_roc, roc1_fpr, roc1_tpr, 'LineWidth', 2.0, 'DisplayName', sprintf('SVM Linear (AUC=%.3f)', roc1_auc));
plot(ax_roc, roc2_fpr, roc2_tpr, 'LineWidth', 2.0, 'DisplayName', sprintf('SVM RBF (AUC=%.3f)', roc2_auc));
plot(ax_roc, roc3_fpr, roc3_tpr, 'LineWidth', 2.0, 'DisplayName', sprintf('k-NN (AUC=%.3f)', roc3_auc));
plot(ax_roc, roc4_fpr, roc4_tpr, 'LineWidth', 2.0, 'DisplayName', sprintf('Decision Tree (AUC=%.3f)', roc4_auc));
plot(ax_roc, [0 1], [0 1], 'k--', 'LineWidth', 1.25, 'DisplayName', 'Random');
xlabel(ax_roc, 'False Positive Rate');
ylabel(ax_roc, 'True Positive Rate');
title(ax_roc, 'Micro-Averaged ROC Curves');
legend(ax_roc, 'Location', 'southeast');
grid(ax_roc, 'on');
axis(ax_roc, 'square');
hold(ax_roc, 'off');

save('classification_action_type_results.mat', 'results', 'class_names_in_use');

fprintf('Done. Plots opened.\n');

% --- Helper functions ---
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

function [acc, sens, spec] = multiclass_metrics(true_y, pred_y, classes)
    acc = mean(true_y == pred_y);
    sens_list = [];
    spec_list = [];

    for c = classes
        tp = sum((true_y == c) & (pred_y == c));
        tn = sum((true_y ~= c) & (pred_y ~= c));
        fp = sum((true_y ~= c) & (pred_y == c));
        fn = sum((true_y == c) & (pred_y ~= c));

        if tp + fn > 0
            sens_list = [sens_list; tp / (tp + fn)]; %#ok<AGROW>
        end
        if tn + fp > 0
            spec_list = [spec_list; tn / (tn + fp)]; %#ok<AGROW>
        end
    end

    sens = mean(sens_list);
    spec = mean(spec_list);
end

function [fpr, tpr, auc] = micro_roc(labels, scores, classes)
    bin_labels = [];
    bin_scores = [];

    for idx = 1:numel(classes)
        c = classes(idx);
        bin_labels = [bin_labels; labels == c]; %#ok<AGROW>
        bin_scores = [bin_scores; scores(:, idx)]; %#ok<AGROW>
    end

    bin_labels = bin_labels(:);
    bin_scores = bin_scores(:);
    [fpr, tpr, ~, auc] = perfcurve(bin_labels, bin_scores, true);
end
