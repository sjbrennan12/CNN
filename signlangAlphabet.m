function [] = signlangAlphabet()
%loads mnist data and trains on 42000 training images using batch back propagation
% and plots performance training and validation set and then calculates 
% error percentage of the test set.

%converts the MNIST data to a matlab format with values between 0-1
%function convertMNIST by Markus Mayer 
training = readtable("train.csv");
training = training{:,:};
test = readtable("test.csv");
test = test{:,:};

number = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%0
          0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%1
          0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%2
          0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%3
          0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%4
          0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%5
          0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%6
          0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%7
          0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%8
          0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%9
          0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%10
          0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;%11
          0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;%12
          0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;%13
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;%14
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;%15
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;%16
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;%17
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;%18
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;%19
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0;%20
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;%21
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;%22
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;%23
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;%24
          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;];%25

%generate multilayer netowrk
backPropNetwork = NetworkBackpropagation(3,2,3,784,26,'logsig');%num layers, input, output
%backPropNetwork.L(1) = BackPropLayer(784,392,'logsig');%specifiy input and output size and transfer function
%backPropNetwork.L(2) = BackPropLayer(392,196,'logsig');
%backPropNetwork.L(3) = BackPropLayer(196,26,'logsig');

performance = 0;%training data mean squared error
verifyPerformance = 0;%validation data mean squared error
result = [];%plot data row1 training, row2 validation
q = 0;%number of samples for means squared error
n = 1;% epoch
figure('name',"Performance Mean Squared Error");
title('Performance');

for i = 1:5
    i
    %select training batch
    in = [];
    out = [];
    perm = randperm(20600);
    %randomoly select validation values
    inVerify = [];
    outVerify = [];
    for count = 1:length(perm)%for all training data
    permVerify = randperm(3427,100) + 20600;%select random verify set values
    
    in(:,1) = training(perm(count),3:786)/255;
    out(:,1) = number(training(perm(count),2)+1,:)';
    inVerify(:,1) = training(permVerify(1),3:786)/255;
    outVerify(:,1) = number(training(permVerify(1),2)+1,:)';
    squareIn = reshape(in,28,28)';
    backPropNetwork = backPropNetwork.calcOutput(squareIn);
    backPropNetwork = backPropNetwork.doBackprop(0.01,out);%perform backprop
    %calculate mean squared error of output for training and verify set
    backPropNetwork = backPropNetwork.calcOutput(squareIn);
    performance = performance + backPropNetwork.meanSquareError(out(:,1));
    backPropNetwork = backPropNetwork.calcOutput(reshape(inVerify,28,28)');
    verifyPerformance = verifyPerformance + backPropNetwork.meanSquareError(outVerify(:,1));
    q = q + 1;
    
    end
    performance/q
    result(1,n) = performance/q;%store training performance for current epoch
    result(2,n) = verifyPerformance/q;%store validation performance for current epoch
    n = n + 1;
end

hold on;
plot(result(1,:));% plot performance of validation and training set
plot(result(2,:));

hold off;
legend('Training Data', 'Validation Data');
%test
categories = table();
categories.id = test(:,1);
temp = [];
finalLayer = length(backPropNetwork.L);

for i =1:7172% for all test data 
backPropNetwork = backPropNetwork.calcOutput(reshape(test(i,2:785),28,28)');
max = backPropNetwork.L(finalLayer).out(1);
number = 0;
for j = 2:26%find max output number 
if(backPropNetwork.L(finalLayer).out(j) > max)
    max = backPropNetwork.L(finalLayer).out(j);
    number = j -1;
end
end
temp(i,1) = number;
end
categories.label = temp;
writetable(categories,'TestOutput.csv');

pass = 0;
for i =24027:27455% for all test data
backPropNetwork = backPropNetwork.calcOutput(reshape(training(i,3:786),28,28)');
max = backPropNetwork.L(finalLayer).out(1);
number = 0;
for j = 2:10%find max output number 
if(backPropNetwork.L(finalLayer).out(j) > max)
    max = backPropNetwork.L(finalLayer).out(j);
    number = j -1;
end
end
if(number == training(i,2))%if result equals the label add to pass
pass = pass +1;
end
end

errorPercent = (1 -(pass/3428)) * 100;
x = ['Test Error Rate = ',num2str(errorPercent),'%'];%print test error rate to console
disp(x);
end


