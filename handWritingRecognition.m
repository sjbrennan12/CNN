function [] = handWritingRecognition()
%E711P2 Summary of this function goes here
%   Detailed explanation goes here
in = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1;%0
      -1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1;%1
      1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1;%2
      ];
out = [1 0 0 ; 0 1 0; 0 0 1]';
result = []; 
backPropNetwork = NetworkBackpropagation(2,30,3,'logsig');
backPropNetwork.L(1) = BackPropLayer(30,15,'logsig');
backPropNetwork.L(2) = BackPropLayer(15,3,'logsig');
performance = 0;
performance4flip = 0;
performance8flip = 0;
q = 0;
n = 1;
figure('name',"Performance");
title('Performance');

for i = 1:100
    for s = 1:10
    backPropNetwork = backPropNetwork.doBatchBackprop(0.9,in',out);
    for j = 1:3
    backPropNetwork = backPropNetwork.calcOutput(in(j,:)');
    performance = performance + backPropNetwork.meanSquareError(out(:,j));
    backPropNetwork = backPropNetwork.calcOutput(addNoise(in(j,:)',4));
    performance4flip = performance4flip + backPropNetwork.meanSquareError(out(:,j));
    backPropNetwork = backPropNetwork.calcOutput(addNoise(in(j,:)',8));
    performance8flip = performance8flip + backPropNetwork.meanSquareError(out(:,j));
    q = q + 1;
    end
    end
    result(1,n) = performance/q;
    result(2,n) = performance4flip/q;
    result(3,n) = performance8flip/q;
    n = n + 1;
end

hold on;
plot(result(1,:));
plot(result(2,:));
plot(result(3,:));
hold off;
legend('0 bits flipped', '4 bits flipped', '8 bits flipped');
disp(performance/q);
disp(backPropNetwork.L(2).out);
backPropNetwork.calcOutput(in(1,:)');

end

