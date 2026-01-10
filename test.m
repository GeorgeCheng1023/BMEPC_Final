files = dir(fullfile('D:\MATLAB\Final project\EMG Physical Action Data Set - del','**','*.txt'));
chCounts = zeros(numel(files),1);
for i=1:numel(files)
    dat = readmatrix(fullfile(files(i).folder,files(i).name));
    if isrow(dat), dat = dat'; end
    if size(dat,1) < size(dat,2), dat = dat'; end
    chCounts(i) = size(dat,2);
end
tabulate(chCounts)