function [] = handWritingRecognition()
%E711P2 Summary of this function goes here
%   Detailed explanation goes here
%built in matlab function to get nnet data

training = convertMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
test = convertMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

number = [1 0 0 0 0 0 0 0 0 0;%0
          0 1 0 0 0 0 0 0 0 0;%1
          0 0 1 0 0 0 0 0 0 0;%2
          0 0 0 1 0 0 0 0 0 0;%3
          0 0 0 0 1 0 0 0 0 0;%4
          0 0 0 0 0 1 0 0 0 0;%5
          0 0 0 0 0 0 1 0 0 0;%6
          0 0 0 0 0 0 0 1 0 0;%7
          0 0 0 0 0 0 0 0 1 0;%8
          0 0 0 0 0 0 0 0 0 1;];%9

backPropNetwork = NetworkBackpropagation(3,784,10,'logsig');
backPropNetwork.L(1) = BackPropLayer(784,392,'logsig');
backPropNetwork.L(2) = BackPropLayer(392,196,'logsig');
backPropNetwork.L(3) = BackPropLayer(196,10,'logsig');

performance = 0;
result = [];
q = 0;
n = 1;
figure('name',"Performance");
title('Performance');

for i = 1:5000
    %randomly select training batch
    in = [];
    out = [];
    perm = randperm(7500,10);
    %randomoly select validation values
    for s = 1:10
    temp = training.images(:,:,perm(s))';
    in(:,s) = temp(:);
    out(:,s) = number(training.labels(perm(s),:)+1,:)';
    end

    backPropNetwork = backPropNetwork.doBatchBackprop(0.7,in,out);
    for j = 1:10
    backPropNetwork = backPropNetwork.calcOutput(in(:,j));
    performance = performance + backPropNetwork.meanSquareError(out(:,j));
    
    q = q + 1;
    end
    result(n) = performance/q;
    n = n + 1;
end

hold on;
plot(result(1,:));

hold off;
%legend('0 bits flipped', '4 bits flipped', '8 bits flipped');
disp(performance/q);


end

