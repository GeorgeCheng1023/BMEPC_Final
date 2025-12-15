clc; clear; close all;

%% ===============================
% 1. Dataset path
%% ===============================
basePath = ...
 'C:\Users\USER\Desktop\醫用電腦\BMEPC_Final\EMG Physical Action Data Set';

subjects = {'sub1','sub2','sub3','sub4'};
conditions = {'Normal','Aggressive'};

X = [];
Y = [];

%% ===============================
% 2. Feature Extraction
%% ===============================
for s = 1:length(subjects)
    for c = 1:length(conditions)
        
        label = (c == 2);  % Normal=0, Aggressive=1
        
        txtPath = fullfile(basePath, subjects{s}, conditions{c}, 'txt');
        files = dir(fullfile(txtPath, '*.txt'));
        
        for f = 1:length(files)
            filePath = fullfile(files(f).folder, files(f).name);
            
            % -------- Read EMG TXT --------
            emg = readmatrix(filePath);  % samples × channels
            
            % -------- Feature Extraction --------
            % RMS
            rmsVal = rms(emg);
            
            % MAV
            mavVal = mean(abs(emg));
            
            % Zero Crossing
            zcVal = sum(diff(sign(emg))~=0);
            
            % Waveform Length
            wlVal = sum(abs(diff(emg)));
            
            featureVector = [rmsVal mavVal zcVal wlVal];
            
            X = [X; featureVector];
            Y = [Y; label];
        end
    end
end

%% ===============================
% 3. Normalization
%% ===============================
X = normalize(X);

%% ===============================
% 4. Train / Validation Split
%% ===============================
rng(1);
cv = cvpartition(Y,'HoldOut',0.2);

XTrain = X(training(cv), :)';
YTrain = Y(training(cv))';

XVal = X(test(cv), :)';
YVal = Y(test(cv))';

YTrain_oh = dummyvar(YTrain+1)';
YVal_oh = dummyvar(YVal+1)';

%% ===============================
% 5. Neural Network Classifier
%% ===============================
hiddenLayerSize = 20;
net = patternnet(hiddenLayerSize);

net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;

net.performFcn = 'crossentropy';

%% ===============================
% 6. Training Progress
%% ===============================
[net, tr] = train(net, XTrain, YTrain_oh);

%% ===============================
% 7. Validation Accuracy
%% ===============================
Ypred = net(XVal);
[~, predictedLabels] = max(Ypred, [], 1);


