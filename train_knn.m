function [acc, sens, spec, scores, labels] = train_knn(Xtr, Ytr, Xte, Yte, varargin)
% TRAIN_KNN Train and test k-Nearest Neighbors classifier
%
% Inputs:
%   Xtr     - Training features (N x D matrix)
%   Ytr     - Training labels (N x 1 vector)
%   Xte     - Test features (M x D matrix)
%   Yte     - Test labels (M x 1 vector)
%   varargin - Optional parameters:
%              'NumNeighbors' - Number of neighbors k (default: 5)
%              'Distance'     - Distance metric (default: 'euclidean')
%
% Outputs:
%   acc    - Accuracy
%   sens   - Sensitivity
%   spec   - Specificity
%   scores - Prediction scores for ROC
%   labels - True labels for ROC

% Parse optional parameters
p = inputParser;
addParameter(p, 'NumNeighbors', 5);
addParameter(p, 'Distance', 'euclidean');
parse(p, varargin{:});

% Train k-NN
model = fitcknn(Xtr, Ytr, ...
    'NumNeighbors', p.Results.NumNeighbors, ...
    'Distance', p.Results.Distance);

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
