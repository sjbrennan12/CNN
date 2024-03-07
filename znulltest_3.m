function [] = test1()
%E711P2 Summary of this function goes here
%   Detailed explanation goes here
in = [-2 ; -1.5; -1; -0.5;0;0.5;1;1.5;2];
out = [0;0.076;0.293;0.617;1;1.383;1.707;1.924;2];
result = [];
backPropNetwork = NetworkBackpropagation(2,1,1,'logsig');
backPropNetwork.L(1) = BackPropLayer(1,2,'logsig');
backPropNetwork.L(2) = BackPropLayer(2,1,'purelin');
result = backPropNetwork.forward(in);
disp(result);
performance = 0;
q = 0;
n = 1;
figure('name',"Performance");
title('Performance');
hold on;
for i = 1:100
    for s = 1:1
    %for j = 1:3
       %backPropNetwork = backPropNetwork.calcOutput(addNoise(in(j,:),0)');
       %performance = performance + backPropNetwork.meanSquareError(out(:,j));
       %q = q + 1;
       %backPropNetwork = backPropNetwork.doBackprop(0.3,out(:,j),0.98);
    %end
    backPropNetwork = backPropNetwork.doBatchBackprop(0.5,in',out');
    performance = performance + backPropNetwork.meanSquareError(out(8,:));
     q = q + 1;
    end
    result(n) = performance/q;
    plot(result);
    n = n + 1;
    
end  
disp(performance/q);
disp(backPropNetwork.L(2).out);

end

