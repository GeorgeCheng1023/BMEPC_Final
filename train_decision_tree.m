function [acc, sens, spec, scores, labels] = train_decision_tree(Xtr, Ytr, Xte, Yte, varargin)
% TRAIN_DECISION_TREE Train and test Decision Tree classifier
%
% Inputs:
%   Xtr     - Training features (N x D matrix)
%   Ytr     - Training labels (N x 1 vector)
%   Xte     - Test features (M x D matrix)
%   Yte     - Test labels (M x 1 vector)
%   varargin - Optional parameters:
%              'MaxNumSplits'   - Maximum number of splits (default: 20)
%              'MinLeafSize'    - Minimum leaf size (default: 1)
%              'Prune'          - Pruning option (default: 'off')
%              'SplitCriterion' - Split criterion (default: 'gdi')
%
% Outputs:
%   acc    - Accuracy
%   sens   - Sensitivity
%   spec   - Specificity
%   scores - Prediction scores for ROC
%   labels - True labels for ROC

% Parse optional parameters
p = inputParser;
addParameter(p, 'MaxNumSplits', 20);
addParameter(p, 'MinLeafSize', 1);
addParameter(p, 'Prune', 'off');
addParameter(p, 'SplitCriterion', 'gdi');
parse(p, varargin{:});

% Train Decision Tree
model = fitctree(Xtr, Ytr, ...
    'MaxNumSplits', p.Results.MaxNumSplits, ...
    'MinLeafSize', p.Results.MinLeafSize, ...
    'Prune', p.Results.Prune, ...
    'SplitCriterion', p.Results.SplitCriterion);

% Predict
[pred, score] = predict(model, Xte);

% Calculate metrics
[acc, sens, spec] = calculate_metrics(Yte, pred);

% Return scores and labels for ROC curve
scores = score(:, 2);
labels = Yte;

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
