clear all
close all
clc
folder='EMG Physical Action Data Set - del';
subs={'sub1','sub2','sub3','sub4'};
normalActions={'Clapping','Handshaking','Hugging','Waving'};
aggressiveActions={'Elbowing','Hamering','Pulling','Punching','Pushing','Slapping'};
ch=8;targetChannels=1:8;
X=[];Y_class=[];Y_normal_action=[];Y_aggr_action=[];

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
for j=1:length(aggressiveActions)
path=fullfile(folder,subs{i},'Aggressive','txt',[aggressiveActions{j},'.txt']);
if ~isfile(path)
continue
end
try
data=readmatrix(path);
if size(data,2)~=ch
data=data';
end
data=data(:,targetChannels);
windowSize=256;stepSize=128;numSamples=size(data,1);
for w=1:stepSize:(numSamples-windowSize+1)
windowData=data(w:w+windowSize-1,:);
feat=extract_features(windowData);
feat_row=reshape(feat',1,[]);
X=[X;feat_row];
Y_class=[Y_class;1];
Y_normal_action=[Y_normal_action;-1];
Y_aggr_action=[Y_aggr_action;j-1];
end
catch
end
end
end
disp('loaded')
disp(size(X,1))
X=zscore(X);
disp('step 1 normal vs aggressive')
folds=5;rng(100);cv1=cvpartition(Y_class,'KFold',folds);
class_acc=[];class_pred_all=[];class_true_all=[];
for f=1:folds
tr=training(cv1,f);te=test(cv1,f);
Xtr=X(tr,:);Ytr=Y_class(tr);Xte=X(te,:);Yte=Y_class(te);
mdl=fitcsvm(Xtr,Ytr,'KernelFunction','rbf','BoxConstraint',100,'KernelScale','auto');
pred=predict(mdl,Xte);
acc=sum(pred==Yte)/length(Yte);
class_acc=[class_acc;acc];
class_pred_all=[class_pred_all;pred];
class_true_all=[class_true_all;Yte];
end

disp('acc:')
disp(mean(class_acc)*100)
disp('step 2 normal actions')
idx_normal=Y_class==0;
X_normal=X(idx_normal,:);Y_normal=Y_normal_action(idx_normal);
rng(100);cv2=cvpartition(Y_normal,'KFold',folds);
normal_acc=[];normal_pred_all=[];normal_true_all=[];
for f=1:folds
tr=training(cv2,f);te=test(cv2,f);
Xtr=X_normal(tr,:);Ytr=Y_normal(tr);Xte=X_normal(te,:);Yte=Y_normal(te);
template=templateTree('MaxNumSplits',50,'MinLeafSize',2,'NumVariablesToSample','all');
mdl=fitcensemble(Xtr,Ytr,'Method','Bag','NumLearningCycles',200,'Learners',template);
pred=predict(mdl,Xte);
acc=sum(pred==Yte)/length(Yte);
normal_acc=[normal_acc;acc];
normal_pred_all=[normal_pred_all;pred];
normal_true_all=[normal_true_all;Yte];
end

disp(mean(normal_acc)*100)
disp('step 3 aggressive actions')
idx_aggr=Y_class==1;
X_aggr=X(idx_aggr,:);Y_aggr=Y_aggr_action(idx_aggr);
rng(100);cv3=cvpartition(Y_aggr,'KFold',folds);
aggr_acc=[];aggr_pred_all=[];aggr_true_all=[];
for f=1:folds
tr=training(cv3,f);te=test(cv3,f);
Xtr=X_aggr(tr,:);Ytr=Y_aggr(tr);Xte=X_aggr(te,:);Yte=Y_aggr(te);
template=templateTree('MaxNumSplits',50,'MinLeafSize',2);
mdl=fitcensemble(Xtr,Ytr,'Method','TotalBoost','NumLearningCycles',200,'Learners',template);
pred=predict(mdl,Xte);
acc=sum(pred==Yte)/length(Yte);
aggr_acc=[aggr_acc;acc];
aggr_pred_all=[aggr_pred_all;pred];
aggr_true_all=[aggr_true_all;Yte];
end

disp(mean(aggr_acc)*100)
disp('summary')
disp(mean(class_acc)*100)
disp(mean(normal_acc)*100)
disp(mean(aggr_acc)*100)

% Print detailed confusion matrices
disp(' ')
disp('=== CONFUSION MATRIX - LEVEL 1: Normal vs Aggressive ===')
cm_class = confusionmat(class_true_all, class_pred_all);
disp('Rows=True, Columns=Predicted (0=Normal, 1=Aggressive)')
disp(cm_class)
disp(' ')

disp('=== CONFUSION MATRIX - LEVEL 2: Normal Actions ===')
cm_norm = confusionmat(normal_true_all, normal_pred_all);
normalActions={'Clapping','Handshaking','Hugging','Waving'};
disp('Rows=True, Columns=Predicted (0-3 for each action)')
disp(cm_norm)
disp(' ')

disp('=== CONFUSION MATRIX - LEVEL 3: Aggressive Actions ===')
cm_agg = confusionmat(aggr_true_all, aggr_pred_all);
aggressiveActions={'Elbowing','Hamering','Pulling','Punching','Pushing','Slapping'};
disp('Rows=True, Columns=Predicted (0-5 for each action)')
disp(cm_agg)
disp(' ')

fig=figure('Name','Hierarchical Classification Results','NumberTitle','off','Position',[100,100,1400,500]);
subplot(1,3,1)
confusionchart(class_true_all,class_pred_all,'Title','Normal vs Aggressive','Normalization','row-normalized')
subplot(1,3,2)
cm_normal=confusionchart(normal_true_all,normal_pred_all,'Title','Normal Actions','Normalization','row-normalized');
cm_normal.RowSummary='row-normalized';
cm_normal.ColumnSummary='column-normalized';
subplot(1,3,3)
cm_aggr=confusionchart(aggr_true_all,aggr_pred_all,'Title','Aggressive Actions','Normalization','row-normalized');
cm_aggr.RowSummary='row-normalized';
cm_aggr.ColumnSummary='column-normalized';
disp('done')

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
