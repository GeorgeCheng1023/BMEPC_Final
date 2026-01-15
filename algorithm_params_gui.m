function algorithm_params_gui()

fig = figure('Name', 'Algorithm Parameters', 'NumberTitle', 'off', 'Position', [100, 100, 600, 500]);

uicontrol('Style', 'text', 'Parent', fig, 'String', 'SVM Linear Parameters', 'Position', [20, 450, 200, 20], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
uicontrol('Style', 'text', 'Parent', fig, 'String', 'Box Constraint:', 'Position', [30, 420, 150, 20], 'HorizontalAlignment', 'left');
svm_lin_box = uicontrol('Style', 'edit', 'Parent', fig, 'String', '10', 'Position', [180, 420, 80, 20]);

uicontrol('Style', 'text', 'Parent', fig, 'String', 'SVM RBF Parameters', 'Position', [20, 370, 200, 20], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
uicontrol('Style', 'text', 'Parent', fig, 'String', 'Box Constraint:', 'Position', [30, 340, 150, 20], 'HorizontalAlignment', 'left');
svm_rbf_box = uicontrol('Style', 'edit', 'Parent', fig, 'String', '50', 'Position', [180, 340, 80, 20]);
uicontrol('Style', 'text', 'Parent', fig, 'String', 'Kernel Scale:', 'Position', [30, 310, 150, 20], 'HorizontalAlignment', 'left');
svm_rbf_kernel = uicontrol('Style', 'edit', 'Parent', fig, 'String', '2', 'Position', [180, 310, 80, 20]);

uicontrol('Style', 'text', 'Parent', fig, 'String', 'k-NN Parameters', 'Position', [20, 260, 200, 20], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
uicontrol('Style', 'text', 'Parent', fig, 'String', 'Number of Neighbors:', 'Position', [30, 230, 150, 20], 'HorizontalAlignment', 'left');
knn_neighbors = uicontrol('Style', 'edit', 'Parent', fig, 'String', '5', 'Position', [180, 230, 80, 20]);

uicontrol('Style', 'text', 'Parent', fig, 'String', 'Decision Tree Parameters', 'Position', [20, 180, 200, 20], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
uicontrol('Style', 'text', 'Parent', fig, 'String', 'Min Leaf Size:', 'Position', [30, 150, 150, 20], 'HorizontalAlignment', 'left');
tree_minleaf = uicontrol('Style', 'edit', 'Parent', fig, 'String', '10', 'Position', [180, 150, 80, 20]);
uicontrol('Style', 'text', 'Parent', fig, 'String', 'Max Num Splits:', 'Position', [30, 120, 150, 20], 'HorizontalAlignment', 'left');
tree_maxsplits = uicontrol('Style', 'edit', 'Parent', fig, 'String', '5', 'Position', [180, 120, 80, 20]);

uicontrol('Style', 'pushbutton', 'Parent', fig, 'String', 'Run Classification', 'Position', [100, 40, 150, 30], 'FontSize', 11, 'Callback', {@run_classification, svm_lin_box, svm_rbf_box, svm_rbf_kernel, knn_neighbors, tree_minleaf, tree_maxsplits});

end

function run_classification(hObject, eventdata, svm_lin_box, svm_rbf_box, svm_rbf_kernel, knn_neighbors, tree_minleaf, tree_maxsplits)

svm_lin_c = str2double(get(svm_lin_box, 'String'));
svm_rbf_c = str2double(get(svm_rbf_box, 'String'));
svm_rbf_ks = str2double(get(svm_rbf_kernel, 'String'));
knn_k = str2double(get(knn_neighbors, 'String'));
tree_ml = str2double(get(tree_minleaf, 'String'));
tree_ms = str2double(get(tree_maxsplits, 'String'));

msgbox('Loading data...', 'Status');

fprintf('Loading data...\n');

folder = 'EMG Physical Action Data Set';
subs = {'sub1', 'sub2', 'sub3', 'sub4'};
types = {'Normal', 'Aggressive'};
ch = 8;  
targetChannels = 1:4; 

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
                
                data = data(:, targetChannels);
                
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

X = zscore(X);

folds = 5;
rng(100);
cv = cvpartition(Y, 'KFold', folds);

fprintf('Training models...\n');

svm_lin_acc = [];
svm_lin_sens = [];
svm_lin_spec = [];

svm_rbf_acc = [];
svm_rbf_sens = [];
svm_rbf_spec = [];

knn_acc = [];
knn_sens = [];
knn_spec = [];

tree_acc = [];
tree_sens = [];
tree_spec = [];

for f = 1:folds
    tr = training(cv, f);
    te = test(cv, f);
    
    Xtr = X(tr, :);
    Ytr = Y(tr);
    Xte = X(te, :);
    Yte = Y(te);
    
    [a1, s1, sp1, ~, ~] = train_svm_linear(Xtr, Ytr, Xte, Yte, 'BoxConstraint', svm_lin_c);
    svm_lin_acc = [svm_lin_acc; a1];
    svm_lin_sens = [svm_lin_sens; s1];
    svm_lin_spec = [svm_lin_spec; sp1];
    
    [a2, s2, sp2, ~, ~] = train_svm_rbf(Xtr, Ytr, Xte, Yte, 'BoxConstraint', svm_rbf_c, 'KernelScale', svm_rbf_ks);
    svm_rbf_acc = [svm_rbf_acc; a2];
    svm_rbf_sens = [svm_rbf_sens; s2];
    svm_rbf_spec = [svm_rbf_spec; sp2];
    
    [a3, s3, sp3, ~, ~] = train_knn(Xtr, Ytr, Xte, Yte, 'NumNeighbors', knn_k);
    knn_acc = [knn_acc; a3];
    knn_sens = [knn_sens; s3];
    knn_spec = [knn_spec; sp3];
    
    [a4, s4, sp4, ~, ~] = train_decision_tree(Xtr, Ytr, Xte, Yte, 'MinLeafSize', tree_ml, 'MaxNumSplits', tree_ms);
    tree_acc = [tree_acc; a4];
    tree_sens = [tree_sens; s4];
    tree_spec = [tree_spec; sp4];
end

fprintf('\n=== RESULTS ===\n\n');

names = {['SVM (Linear, C=', num2str(svm_lin_c), ')'] ...
    ; ['SVM (RBF, C=', num2str(svm_rbf_c), ')'] ...
    ; ['k-NN (k=', num2str(knn_k), ')'] ...
    ; ['Decision Tree (ML=', num2str(tree_ml), ')']};

acc = [mean(svm_lin_acc)*100; mean(svm_rbf_acc)*100; mean(knn_acc)*100; mean(tree_acc)*100];
sens = [mean(svm_lin_sens)*100; mean(svm_rbf_sens)*100; mean(knn_sens)*100; mean(tree_sens)*100];
spec = [mean(svm_lin_spec)*100; mean(svm_rbf_spec)*100; mean(knn_spec)*100; mean(tree_spec)*100];

tbl = table(names, acc, sens, spec, 'VariableNames', {'Model', 'Accuracy', 'Sensitivity', 'Specificity'});
disp(tbl);

[best_val, best_idx] = max(acc);
fprintf('\nBest: %s (%.2f%%)\n\n', names{best_idx}, best_val);

msgbox(sprintf('Classification Complete!\n\nBest Model: %s\nAccuracy: %.2f%%', names{best_idx}, best_val), 'Results');

fprintf('Creating plots...\n');

fig = figure('Name', 'EMG Results', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 700]);
tg = uitabgroup(fig);

tab1 = uitab(tg, 'Title', 'Confusion Matrices');

m1_all = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', svm_lin_c);
p1_all = predict(m1_all, X);

m2_all = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'Standardize', false, 'BoxConstraint', svm_rbf_c);
p2_all = predict(m2_all, X);

m3_all = fitcknn(X, Y, 'NumNeighbors', knn_k, 'Distance', 'euclidean');
p3_all = predict(m3_all, X);

m4_all = fitctree(X, Y, 'MinLeafSize', tree_ml);
p4_all = predict(m4_all, X);

subplot(2, 2, 1, 'Parent', tab1);
confusionchart(Y, p1_all, 'Title', 'SVM (Linear)', 'Normalization', 'row-normalized');

subplot(2, 2, 2, 'Parent', tab1);
confusionchart(Y, p2_all, 'Title', 'SVM (RBF)', 'Normalization', 'row-normalized');

subplot(2, 2, 3, 'Parent', tab1);
confusionchart(Y, p3_all, 'Title', 'k-NN', 'Normalization', 'row-normalized');

subplot(2, 2, 4, 'Parent', tab1);
confusionchart(Y, p4_all, 'Title', 'Decision Tree', 'Normalization', 'row-normalized');

tab3 = uitab(tg, 'Title', 'Metrics');
ax_met = axes('Parent', tab3);
metrics_mat = [acc sens spec];
bar(ax_met, metrics_mat, 'LineWidth', 1.1);
set(ax_met, 'XTickLabel', names, 'XTickLabelRotation', 15, 'FontSize', 11);
legend(ax_met, {'Accuracy', 'Sensitivity', 'Specificity'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
ylabel(ax_met, 'Percentage (%)', 'FontSize', 12);
title(ax_met, 'Cross-Validation Metrics', 'FontSize', 14);
ylim(ax_met, [0 100]);
grid(ax_met, 'on');

tab4 = uitab(tg, 'Title', 'Decision Boundary');
ax_db = axes('Parent', tab4);
hold(ax_db, 'on');

m_lin = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', svm_lin_c);

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

end

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
