classdef NetworkBackpropagation
    %NetworkBackpropagation is responsible for the operations on the whole
    %neural network and will calculate the forward output, and complete
    %backpropagation
    
    properties
      L BackPropLayer;%holds alls the layers for the network 
    end
    
    methods
        function obj = NetworkBackpropagation(numLayers,inputs, outputs,transferFunction)
            %constructor that will set the initial number of layers and
            %weight and bias sizes. Initially the layers will have weights
            %and biases that are the same example a network with 20 inputs
            %and 2 outputs and 2 layers will have a network that looks like
            %20 20 2
           for i = 1:(numLayers -1) 
            obj.L(i) = BackPropLayer(inputs,inputs,transferFunction);%construct all layers except final layer
           end
            obj.L(numLayers) = BackPropLayer(inputs,outputs,transferFunction);%final layer output matches the provided output
            
        end
        
        function obj = calcOutput(obj,input)
            %takes the input and passes it through each layer and moves
            %the output through the remaining layers until the end
        nextLayer = input;%input into the layer
        for i = 1:length(obj.L)
        obj.L(i) = obj.L(i).forward(nextLayer);% find the output of the layer from the input
        nextLayer = obj.L(i).out;%save the output for the next layer as an input
        end
        %display(nextLayer);
        end

        function obj = doBatchBackprop(obj,learningRate,inputs,targets)
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
        
        function obj = calculate_sensitivities(obj, output, target, i)
            % calculate_sensitivities
            % Perform an iteration of the backpropagation algorithm.
            % Each layer saves the sensitivities of the weights.
            % Weights are not updated here; use the update function in the Layer class.
        
            error = Error(target, output);
            
            % Compute sensitivity for the last layer
            result = output .* (1 - output);
            F_dot = diag(result);
            sens = -2 * F_dot * error;
            obj.L(end).Sensitivity = sens;
        
            % Backpropagate sensitivity through the layers
            for j = length(obj.L) - 1:-1:i
                nextSens = obj.L(j + 1).Sensitivity;
                currAct = obj.L(j).Output;
                p_derv = currAct .* (1 - currAct);
                F_dot = diag(p_derv);
                sens = F_dot * obj.L(j + 1).Weights' * nextSens;
                obj.L(j).Sensitivity = sens;
            end
        end


         function error = Error(~,t,a)
            % error
            % calculates the error of:
            % the target vector T
            % the actual vector a
               error = t-a;
         end

       
        end

        function output = forward(obj, input)
            % Apply convolution operation using the first layer
            output = conv2(input, obj.L(1).weight, 'valid');
            
            % Apply ReLU activation
            output = max(output, 0);
        end

        function obj = doBackprop(obj,learningRate, input, target)%single input output backprop
            output = forward(obj, input, target);

            %calculate sensitivities 
            obj.L(length(obj.L)) = obj.L(length(obj.L)).firstSensitivity(target);%calculate first sensitivity 
            for i = (length(obj.L) - 1):-1:1%calculate all remaining sensitivities
                obj.L(i) = calculate_sensitivities(obj, output, target, i);    
            end
            
            %Calculate new weights and biases
            for i = length(obj.L):-1:2
                obj.L(i) = obj.L(i).newWeight(learningRate,obj.L(i -1).out); 
                obj.L(i) = obj.L(i).newBias(learningRate);
            end
            obj.L(1) = obj.L(1).newWeight(learningRate,obj.L(1).in); 
            obj.L(1) = obj.L(1).newBias(learningRate);
            
    
            %set new weights and biases
            for i = length(obj.L):-1:1
                obj.L(i) = obj.L(i).setWeightBias(obj.L(i).w2,obj.L(i).b2);
            end
        end
        
        function E = meanSquareError(obj,target)
        a = obj.L(length(obj.L)).out;%calculate the forward output
        E = (target - a)' * (target - a);%mean squared error
        end
    end
end

