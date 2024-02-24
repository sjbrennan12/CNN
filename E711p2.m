function [] = E711p2()
%Implements multilayer network for part 1 by using BackPropLayer and
%NetworkBackpropagation classes and displays the performance of the network
%as it is being trained. Noise performance is also shown on this graph by
%displaying the mean squared error of 4 and 8 bits randomly flipped.
in = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1;%0
      -1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1;%1
      1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1;%2
      ];
out = [1 0 0 ; 0 1 0; 0 0 1]';
result = []; %stores mean squared error that is graphed. row1 = no noise, row2 = 4 bits flipped, row3 = 8 bits flipped, n = epoch
backPropNetwork = NetworkBackpropagation(2,30,3,'logsig');% generate a network that is 2 layer 30 15 3
backPropNetwork.L(1) = BackPropLayer(30,15,'logsig');
backPropNetwork.L(2) = BackPropLayer(15,3,'logsig');
performance = 0;% mean squared error per epoch
performance4flip = 0;
performance8flip = 0;
q = 0;%number of samples
n = 1;%epoch
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
plot(result(1,:));%generate performance graph
plot(result(2,:));
plot(result(3,:));
hold off;
legend('0 bits flipped', '4 bits flipped', '8 bits flipped');
disp(performance/q);% print final error value 
disp(backPropNetwork.L(2).out);% show 8 bit flipped final output
backPropNetwork.calcOutput(in(1,:)');

figure();%display layer 1 weight
imagesc(backPropNetwork.L(1).weight);
colormap(hsv);
colorbar;

figure();%display layer 2 weight
imagesc(backPropNetwork.L(2).weight);
colormap(hsv);
colorbar;

end

