function [acc, sens, spec, scores, labels] = train_svm_linear(Xtr, Ytr, Xte, Yte, varargin)
% TRAIN_SVM_LINEAR Train and test SVM with linear kernel
%
% Inputs:
%   Xtr     - Training features (N x D matrix)
%   Ytr     - Training labels (N x 1 vector)
%   Xte     - Test features (M x D matrix)
%   Yte     - Test labels (M x 1 vector)
%   varargin - Optional parameters:
%              'BoxConstraint' - Regularization (default: 1)
%              'Standardize'   - Standardize features (default: false)
%
% Outputs:
%   acc    - Accuracy
%   sens   - Sensitivity
%   spec   - Specificity
%   scores - Prediction scores for ROC
%   labels - True labels for ROC

% Parse optional parameters
p = inputParser;
addParameter(p, 'BoxConstraint', 1);
addParameter(p, 'Standardize', false);
parse(p, varargin{:});

% Train SVM Linear
model = fitcsvm(Xtr, Ytr, ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', p.Results.BoxConstraint, ...
    'Standardize', p.Results.Standardize);

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
