clear all
close all
clc
folder='EMG Physical Action Data Set';
subs={'sub1','sub2','sub3','sub4'};
types={'Normal','Aggressive'};
ch=8;
targetChannels=1:4;
X=[];Y=[];
for i=1:length(subs)
for j=1:length(types)
path=fullfile(folder,subs{i},types{j},'txt');
if ~isfolder(path)
continue
end
files=dir(fullfile(path,'*.txt'));
for k=1:length(files)
try
data=readmatrix(fullfile(path,files(k).name));
if size(data,2)~=ch
data=data';
end
data=data(:,targetChannels);
feat=extract_features(data);
feat_row=reshape(feat',1,[]);
X=[X;feat_row];
Y=[Y;j-1];
catch
end
end
end
end
disp('loaded')
disp(size(X,1))
X=zscore(X);
folds=5;rng(100);cv=cvpartition(Y,'KFold',folds);
disp('testing svm linear')
box_vals=[0.1,0.5,1,5,10,50,100];
svm_lin_results=zeros(length(box_vals),3);
for i=1:length(box_vals)
acc_temp=[];sens_temp=[];spec_temp=[];
for f=1:folds
tr=training(cv,f);te=test(cv,f);
Xtr=X(tr,:);Ytr=Y(tr);Xte=X(te,:);Yte=Y(te);
[a,s,sp,~,~]=train_svm_linear(Xtr,Ytr,Xte,Yte,'BoxConstraint',box_vals(i));
acc_temp=[acc_temp;a];sens_temp=[sens_temp;s];spec_temp=[spec_temp;sp];
end
svm_lin_results(i,1)=mean(acc_temp)*100;
svm_lin_results(i,2)=mean(sens_temp)*100;
svm_lin_results(i,3)=mean(spec_temp)*100;
end

disp('svm rbf box')
box_vals_rbf=[0.1,0.5,1,5,10,50,100];
svm_rbf_results=zeros(length(box_vals_rbf),3);
for i=1:length(box_vals_rbf)
acc_temp=[];sens_temp=[];spec_temp=[];
for f=1:folds
tr=training(cv,f);te=test(cv,f);
Xtr=X(tr,:);Ytr=Y(tr);Xte=X(te,:);Yte=Y(te);
[a,s,sp,~,~]=train_svm_rbf(Xtr,Ytr,Xte,Yte,'BoxConstraint',box_vals_rbf(i));
acc_temp=[acc_temp;a];sens_temp=[sens_temp;s];spec_temp=[spec_temp;sp];
end
svm_rbf_results(i,1)=mean(acc_temp)*100;
svm_rbf_results(i,2)=mean(sens_temp)*100;
svm_rbf_results(i,3)=mean(spec_temp)*100;
end

disp('kernel scale')
kernel_vals={0.1,0.5,1,2,5,10,'auto'};
svm_rbf_kernel_results=zeros(length(kernel_vals),3);
for i=1:length(kernel_vals)
acc_temp=[];sens_temp=[];spec_temp=[];
for f=1:folds
tr=training(cv,f);te=test(cv,f);
Xtr=X(tr,:);Ytr=Y(tr);Xte=X(te,:);Yte=Y(te);
[a,s,sp,~,~]=train_svm_rbf(Xtr,Ytr,Xte,Yte,'KernelScale',kernel_vals{i});
acc_temp=[acc_temp;a];sens_temp=[sens_temp;s];spec_temp=[spec_temp;sp];
end
svm_rbf_kernel_results(i,1)=mean(acc_temp)*100;
svm_rbf_kernel_results(i,2)=mean(sens_temp)*100;
svm_rbf_kernel_results(i,3)=mean(spec_temp)*100;
end

disp('knn')
k_vals=[1,3,5,7,9,11,15,20];
knn_results=zeros(length(k_vals),3);
for i=1:length(k_vals)
acc_temp=[];sens_temp=[];spec_temp=[];
for f=1:folds
tr=training(cv,f);te=test(cv,f);
Xtr=X(tr,:);Ytr=Y(tr);Xte=X(te,:);Yte=Y(te);
[a,s,sp,~,~]=train_knn(Xtr,Ytr,Xte,Yte,'NumNeighbors',k_vals(i));
acc_temp=[acc_temp;a];sens_temp=[sens_temp;s];spec_temp=[spec_temp;sp];
end
knn_results(i,1)=mean(acc_temp)*100;
knn_results(i,2)=mean(sens_temp)*100;
knn_results(i,3)=mean(spec_temp)*100;
end

disp('tree splits')
split_vals=[5,10,20,30,40,50,100];
tree_results=zeros(length(split_vals),3);
for i=1:length(split_vals)
acc_temp=[];sens_temp=[];spec_temp=[];
for f=1:folds
tr=training(cv,f);te=test(cv,f);
Xtr=X(tr,:);Ytr=Y(tr);Xte=X(te,:);Yte=Y(te);
[a,s,sp,~,~]=train_decision_tree(Xtr,Ytr,Xte,Yte,'MaxNumSplits',split_vals(i));
acc_temp=[acc_temp;a];sens_temp=[sens_temp;s];spec_temp=[spec_temp;sp];
end
tree_results(i,1)=mean(acc_temp)*100;
tree_results(i,2)=mean(sens_temp)*100;
tree_results(i,3)=mean(spec_temp)*100;
end

disp('tree leaf')
leaf_vals=[1,5,10,15,20,25,30];
tree_leaf_results=zeros(length(leaf_vals),3);
for i=1:length(leaf_vals)
acc_temp=[];sens_temp=[];spec_temp=[];
for f=1:folds
tr=training(cv,f);te=test(cv,f);
Xtr=X(tr,:);Ytr=Y(tr);Xte=X(te,:);Yte=Y(te);
[a,s,sp,~,~]=train_decision_tree(Xtr,Ytr,Xte,Yte,'MinLeafSize',leaf_vals(i));
acc_temp=[acc_temp;a];sens_temp=[sens_temp;s];spec_temp=[spec_temp;sp];
end
tree_leaf_results(i,1)=mean(acc_temp)*100;
tree_leaf_results(i,2)=mean(sens_temp)*100;
tree_leaf_results(i,3)=mean(spec_temp)*100;
end

disp('plots')
fig=figure('Name','Parameter Comparison','NumberTitle','off','Position',[50,50,1400,900]);
subplot(3,3,1)
plot(box_vals,svm_lin_results(:,1),'b-o','LineWidth',2,'MarkerSize',8)
xlabel('BoxConstraint')
ylabel('Accuracy (%)')
title('SVM Linear - BoxConstraint')
grid on
set(gca,'XScale','log')

subplot(3,3,2)
plot(box_vals,svm_lin_results(:,2),'r-o','LineWidth',2,'MarkerSize',8)
hold on
plot(box_vals,svm_lin_results(:,3),'g-o','LineWidth',2,'MarkerSize',8)
xlabel('BoxConstraint')
ylabel('Rate (%)')
title('SVM Linear - Sensitivity & Specificity')
legend('Sensitivity','Specificity','Location','best')
grid on
set(gca,'XScale','log')
subplot(3,3,3)
plot(box_vals_rbf,svm_rbf_results(:,1),'b-s','LineWidth',2,'MarkerSize',8)
xlabel('BoxConstraint')
ylabel('Accuracy (%)')
title('SVM RBF - BoxConstraint')
grid on
set(gca,'XScale','log')
subplot(3,3,4)
numeric_kernel=[0.1,0.5,1,2,5,10];
plot(numeric_kernel,svm_rbf_kernel_results(1:6,1),'b-s','LineWidth',2,'MarkerSize',8)
xlabel('KernelScale')
ylabel('Accuracy (%)')
title('SVM RBF - KernelScale')
grid on
set(gca,'XScale','log')

subplot(3,3,5)
plot(k_vals,knn_results(:,1),'b-d','LineWidth',2,'MarkerSize',8)
xlabel('Number of Neighbors (k)')
ylabel('Accuracy (%)')
title('k-NN - NumNeighbors')
grid on
subplot(3,3,6)
plot(k_vals,knn_results(:,2),'r-d','LineWidth',2,'MarkerSize',8)
hold on
plot(k_vals,knn_results(:,3),'g-d','LineWidth',2,'MarkerSize',8)
xlabel('Number of Neighbors (k)')
ylabel('Rate (%)')
title('k-NN - Sensitivity & Specificity')
legend('Sensitivity','Specificity','Location','best')
grid on
subplot(3,3,7)
plot(split_vals,tree_results(:,1),'b-^','LineWidth',2,'MarkerSize',8)
xlabel('MaxNumSplits')
ylabel('Accuracy (%)')
title('Decision Tree - MaxNumSplits')
grid on
subplot(3,3,8)
plot(leaf_vals,tree_leaf_results(:,1),'b-v','LineWidth',2,'MarkerSize',8)
xlabel('MinLeafSize')
ylabel('Accuracy (%)')
title('Decision Tree - MinLeafSize')
grid on
subplot(3,3,9)
[~,best_svm_lin]=max(svm_lin_results(:,1));
[~,best_svm_rbf]=max(svm_rbf_results(:,1));
[~,best_knn]=max(knn_results(:,1));
[~,best_tree_split]=max(tree_results(:,1));
best_accs=[svm_lin_results(best_svm_lin,1),svm_rbf_results(best_svm_rbf,1),knn_results(best_knn,1),tree_results(best_tree_split,1)];
bar(best_accs)
set(gca,'XTickLabel',{'SVM Lin','SVM RBF','k-NN','Tree'})
ylabel('Best Accuracy (%)')
title('Best Performance Comparison')
grid on
ylim([min(best_accs)-5,max(best_accs)+5])

disp('best results')
[best_acc_lin,idx_lin]=max(svm_lin_results(:,1));
disp('SVM Linear')
disp(box_vals(idx_lin))
disp(best_acc_lin)

[best_acc_rbf,idx_rbf]=max(svm_rbf_results(:,1));
[best_acc_kern,idx_kern]=max(svm_rbf_kernel_results(:,1));
[best_acc_knn,idx_knn]=max(knn_results(:,1));
[best_acc_tree,idx_tree]=max(tree_results(:,1));
[best_acc_leaf,idx_leaf]=max(tree_leaf_results(:,1));
disp('done')
save('parameter_comparison_results.mat','svm_lin_results','svm_rbf_results','svm_rbf_kernel_results','knn_results','tree_results','tree_leaf_results','box_vals','box_vals_rbf','kernel_vals','k_vals','split_vals','leaf_vals')
function feat=extract_features(data)
n=size(data,2);
feat=zeros(3,n);
for i=1:n
sig=data(:,i);
feat(1,i)=rms(sig);
feat(2,i)=sum(abs(diff(sig)));
feat(3,i)=var(sig);
end
end
