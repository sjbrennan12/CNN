layer1 = ConvolutionLayer1(2, 3, 1, 2, "relu");
layer2 = ConvolutionLayer1(3, 3, 2, 2, "relu");

input_image = 2* rand(20, 20, 1)-1;

obj = layer1.forward(input_image);
obj2 = layer2.forward(obj.out);

%dummy sensitivities and weights of FC layer
sensitivities = 2 * rand(1, 24) - 1;
sensitivities = sensitivities';
weights = 2 * rand(24, 48) - 1;

obj2 = firstSensitivity(obj2, sensitivities, weights);
disp(obj)
disp(obj2)

obj = updateSensitivity(obj, obj2.s, obj2.Kernels);
