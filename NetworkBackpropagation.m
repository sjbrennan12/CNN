classdef NetworkBackpropagation
    %NetworkBackpropagation is responsible for the operations on the whole
    %neural network and will calculate the forward output, and complete
    %backpropagation
    % Methods:
    %    - NetworkBackpropagation (constructor)
    %    - flattenOutput
    %    - calcOutput
    %    - doBatchBackprop
    %    - forward
    %    - doBackprop
    %    - meanSquareError

    properties
      L BackPropLayer;%holds alls the layers for the network 
      C ConvolutionLayer; %holds all the convolution layers for the network
    end
    
    methods
        function obj = NetworkBackpropagation(numConnectedLayers,numConvolutionLayers, KernelSize, inputs, outputs,transferFunction)
            %constructor that will set the initial number of layers and
            %weight and bias sizes. Initially the layers will have weights
            %and biases that are the same example a network with 20 inputs
            %and 2 outputs and 2 layers will have a network that looks like
            %20 20 2

           currentDepth = 1;
           for i =1:(numConvolutionLayers)
           obj.C(i) = ConvolutionLayer(i*4,KernelSize,KernelSize,currentDepth,2,'relu');
           currentDepth = i*4;
           end
           InputSize = currentDepth * (inputs/4*length(obj.C));%calculate size of input after convolution layer 
           % #outputs * size of outputs at end of convolution layers
           for i = 1:(numConnectedLayers -1) 
            obj.L(i) = BackPropLayer(InputSize,InputSize,transferFunction);%construct all layers except final layer
           end
            obj.L(numLayers) = BackPropLayer(InputSize,outputs,transferFunction);%final layer output matches the provided output
            
        end
        % end of constructor

        function output = flattenOutput(obj,input)
        %3d input from convolutional neural network to vector
        n = 1;
        output = [];
        [row,col,depth] = size(input);
        for i = 1:depth
            for j = 1:row
                for k = 1:col
                output(n) = input(j,k,i);
                n = n +1;
                end
            end
        end
        end
        %end of flattenOutput

        function obj = calcOutput(obj,input)
        %takes the input and passes it through each layer and moves
        %the output through the remaining layers until the end
        nextLayer = input;%input into the layer
        for i = 1:length(obj.C)%first input into convolutional layer
        obj.C(i) = obj.C(i).forward(nextLayer);
        nextLayer = obj.L(i).out;
        end
        nextLayer = flattenOutput(nextLayer);%flatten the output to the fully connected layer
        for i = 1:length(obj.L)
        obj.L(i) = obj.L(i).forward(nextLayer);% find the output of the layer from the input
        nextLayer = obj.L(i).out;%save the output for the next layer as an input
        end
        %display(nextLayer);
        end
        %end of calcOutput

        %batch propagation function that takes a matrix of inputs and
        %targets and calculates the next weight and bias values
        function obj = doBatchBackprop(obj,learningRate,inputs,targets)%todo if we have time
        %batch propagation function that takes a matrix of inputs and
        %targets and calculates the next weight and bias values
        [row,col] = size(inputs);%used to determine the size of multiple loops
        %this section is run first to initialize the AS and allS size
        %before processing the other inputs
         obj = obj.calcOutput(inputs(:,1));% start process by calculating outputs
         obj.L(length(obj.L)) = obj.L(length(obj.L)).firstSensitivity(targets(:,1));%find all layer sensitivities
         for i = (length(obj.L) - 1):-1:1
         obj.L(i) = obj.L(i).Sensitivity(obj.L(i+1).s,obj.L(i+1).weight);    
         end

         for i = length(obj.L):-1:2%calculate all gradients and sensitivites for each layer and save them in the total
         obj.L(i).AS = obj.L(i).calcGradient(obj.L(i-1).out); 
         obj.L(i).allS = obj.L(i).s;
         end
         obj.L(1).AS = obj.L(1).calcGradient(obj.L(1).in); 
         obj.L(1).allS = obj.L(1).s;
        for p = 2:col% for the remaining inputs add sensitivities and gradients to total
            obj = obj.calcOutput(inputs(:,p));
            %calculate sensitivities and outputs
            obj.L(length(obj.L)) = obj.L(length(obj.L)).firstSensitivity(targets(:,p));
            for i = (length(obj.L) - 1):-1:1
                obj.L(i) = obj.L(i).Sensitivity(obj.L(i+1).s,obj.L(i+1).weight);    
            end
            for i = length(obj.L):-1:2
            obj.L(i).AS = obj.L(i).AS + obj.L(i).calcGradient(obj.L(i-1).out); 
            obj.L(i).allS = obj.L(i).allS + obj.L(i).s;
            end
            obj.L(1).AS = obj.L(1).AS + obj.L(1).calcGradient(obj.L(1).in); 
            obj.L(1).allS = obj.L(1).allS + obj.L(1).s;      
        end 
        %Calculate new weights and biases from batch and set values
        for i = 1:length(obj.L)% calculate a new batch weight and bias for every layer 
        obj.L(i) = obj.L(i).newBatchWeight(learningRate,col); 
        obj.L(i) = obj.L(i).newBatchBias(learningRate,col);
        end
        end
        %end of batchBackprop

        function obj = doBackprop(obj,learningRate,target)%single input output backprop
        %calculate sensitivities 
        obj.L(length(obj.L)) = obj.L(length(obj.L)).firstSensitivity(target);%calculate first sensitivity 
        for i = (length(obj.L) - 1):-1:1%calculate all remaining sensitivities
        obj.L(i) = obj.L(i).Sensitivity(obj.L(i+1).s,obj.L(i+1).weight);    
        end
        %calculate sensitivities for the convolutional layer
        obj.C(length(obj.C)) = obj.C(length(obj.L)).vectorSensitivity(obj.L(1).s,obj.L(1).weight);
       
        for i = (length(obj.C) - 1):-1:1%calculate all remaining sensitivities for convolutional layers
        obj.C(i) = obj.C(i).Sensitivity(obj.C(i+1).s,obj.C(i+1).Kernels);    
        end
        %Calculate new weights and biases
        for i = length(obj.L):-1:2
        obj.L(i) = obj.L(i).newWeight(learningRate,obj.L(i -1).out); 
        obj.L(i) = obj.L(i).newBias(learningRate);
        end
        obj.L(1) = obj.L(1).newWeight(learningRate,obj.L(1).in); 
        obj.L(1) = obj.L(1).newBias(learningRate);
        for i = length(obj.C):-1:2
        obj.C(i) = obj.C(i).newWeight(learningRate,obj.C(i -1).out); 
        obj.C(i) = obj.C(i).newBias(learningRate);
        end
        obj.C(1) = obj.C(1).newWeight(learningRate,obj.C(1).in); 
        obj.C(1) = obj.C(1).newBias(learningRate);
        %set new weights and biases
        for i = length(obj.L):-1:1
        obj.L(i) = obj.L(i).setWeightBias(obj.L(i).Kernel2,obj.L(i).b2);
        end
        for i = length(obj.C):-1:1
        obj.C(i) = obj.C(i).setWeightBias(obj.C(i).Kernel2,obj.C(i).b2);
        end
        end
        %end of do Backprop

        function E = meanSquareError(obj,target)
        a = obj.L(length(obj.L)).out;%calculate the forward output
        E = (target - a)' * (target - a);%mean squared error
        end
        % end of meanSquareError function
    end
    % end of public methods
end
% end of NetworkBackpropagation class
