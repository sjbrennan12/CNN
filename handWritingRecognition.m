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
backPropNetwork.L(1) = BackPropLayer(784,262,'logsig');
backPropNetwork.L(2) = BackPropLayer(262,88,'logsig');
backPropNetwork.L(3) = BackPropLayer(88,10,'logsig');

performance = 0;
verifyPerformance = 0;
result = [];
q = 0;
n = 1;
figure('name',"Performance Mean Squared Error");
title('Performance');

for i = 1:15000
    %randomly select training batch
    in = [];
    out = [];
    perm = randperm(7500,10);
    %randomoly select validation values
    inVerify = [];
    outVerify = [];
    permVerify = randperm(2500,10) + 7500;

    for s = 1:10 
    temp = training.images(:,:,perm(s))';
    in(:,s) = temp(:);
    out(:,s) = number(training.labels(perm(s),:)+1,:)';
    temp = training.images(:,:,permVerify(s))';
    inVerify(:,s) = temp(:);
    outVerify(:,s) = number(training.labels(permVerify(s),:)+1,:)';
    end

    backPropNetwork = backPropNetwork.doBatchBackprop(0.7,in,out);
    for j = 1:10
    backPropNetwork = backPropNetwork.calcOutput(in(:,j));
    performance = performance + backPropNetwork.meanSquareError(out(:,j));
    backPropNetwork = backPropNetwork.calcOutput(inVerify(:,j));
    verifyPerformance = verifyPerformance + backPropNetwork.meanSquareError(outVerify(:,j));
    q = q + 1;
    end
    result(1,n) = performance/q;
    result(2,n) = verifyPerformance/q;
    n = n + 1;
end

hold on;
plot(result(1,:));
plot(result(2,:));

hold off;
legend('Training Data', 'Validation Data');
%accuracy test with test data
pass = 0;
finalLayer = length(backPropNetwork.L);

for i =1:10000
temp = test.images(:,:,i)'; 
backPropNetwork = backPropNetwork.calcOutput(temp(:));
max = backPropNetwork.L(finalLayer).out(1);
number = 0;
for j = 2:10
if(backPropNetwork.L(finalLayer).out(j) > max)
    max = backPropNetwork.L(finalLayer).out(j);
    number = j -1;
end
end
if(number == test.labels(i))
pass = pass +1;
end
end
%legend('0 bits flipped', '4 bits flipped', '8 bits flipped');
disp(performance/q);
errorPercent = (1 -(pass/10000)) * 100;
x = ['Test Error Rate = ',num2str(errorPercent),'%'];
disp(x);

for i =1:finalLayer
figure();
imagesc(backPropNetwork.L(i).weight);
colormap(hsv);
colorbar;
end
end

