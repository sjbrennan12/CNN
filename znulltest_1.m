function [] = handWritingRecognition()
%loads mnist data and trains on 42000 training images using batch back propagation
% and plots performance training and validation set and then calculates 
% error percentage of the test set.

%converts the MNIST data to a matlab format with values between 0-1
%function convertMNIST by Markus Mayer 
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

%generate multilayer netowrk
backPropNetwork = NetworkBackpropagation(2,784,10,'logsig');%num layers, input, output
backPropNetwork.L(1) = BackPropLayer(784,262,'logsig');%specifiy input and output size and transfer function
backPropNetwork.L(2) = BackPropLayer(262,10,'logsig');
%backPropNetwork.L(3) = BackPropLayer(88,10,'logsig');
%backPropNetwork.L(4) = BackPropLayer(30,10,'logsig');
performance = 0;%trainign data mean squared error
verifyPerformance = 0;%validation data mean squared error
result = [];%plot data row1 training, row2 validation
q = 0;%number of samples for means quared error
n = 1;% epoch
figure('name',"Performance Mean Squared Error");
title('Performance');

for i = 1:10
    %select training batch
    in = [];
    out = [];
    perm = randperm(42000);
    %randomoly select validation values
    inVerify = [];
    outVerify = [];
    for count = 1:10:(length(perm) - 10);%for all training data
    permVerify = randperm(18000,10) + 42000;%select random verify set values
    for s = 1:10 %load training data and verify values into batches
    temp = training.images(:,:,perm(count+s))';
    in(:,s) = temp(:);
    out(:,s) = number(training.labels(perm(count +s),:)+1,:)';
    temp = training.images(:,:,permVerify(s))';
    inVerify(:,s) = temp(:);
    outVerify(:,s) = number(training.labels(permVerify(s),:)+1,:)';
    end

    backPropNetwork = backPropNetwork.doBatchBackprop(0.7,in,out);%perform batch backprop
    for j = 1:10%calculate mean squared error of output for training and verify set
    backPropNetwork = backPropNetwork.calcOutput(in(:,j));
    performance = performance + backPropNetwork.meanSquareError(out(:,j));
    backPropNetwork = backPropNetwork.calcOutput(inVerify(:,j));
    verifyPerformance = verifyPerformance + backPropNetwork.meanSquareError(outVerify(:,j));
    q = q + 1;
    end 
    end
    result(1,n) = performance/q;%store training performance for current epoch
    result(2,n) = verifyPerformance/q;%store validation performance for current epoch
    n = n + 1;
end

hold on;
plot(result(1,:));% plot performance of validation and training set
plot(result(2,:));

hold off;
legend('Training Data', 'Validation Data');
%accuracy test with test data
pass = 0;
finalLayer = length(backPropNetwork.L);

for i =1:10000% for all test data
temp = test.images(:,:,i)'; 
backPropNetwork = backPropNetwork.calcOutput(temp(:));
max = backPropNetwork.L(finalLayer).out(1);
number = 0;
for j = 2:10%find max output number 
if(backPropNetwork.L(finalLayer).out(j) > max)
    max = backPropNetwork.L(finalLayer).out(j);
    number = j -1;
end
end
if(number == test.labels(i))%if result equals the label add to pass
pass = pass +1;
end
end
%legend('0 bits flipped', '4 bits flipped', '8 bits flipped');
disp(performance/q);
errorPercent = (1 -(pass/10000)) * 100;
x = ['Test Error Rate = ',num2str(errorPercent),'%'];%print test error rate to console
disp(x);

for i =1:finalLayer%print all layer weights
figure();
imagesc(backPropNetwork.L(i).weight);
colormap(hsv);
colorbar;
end
end

