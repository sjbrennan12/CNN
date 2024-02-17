classdef NetworkBackpropagation
    %NETWORKBACKPROPAGATION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
      L BackPropLayer;
    end
    
    methods
        function obj = NetworkBackpropagation(numLayers,inputs, outputs,transferFunction)
           for i = 1:(numLayers -1) 
            obj.L(i) = BackPropLayer(inputs,inputs,transferFunction);
           end
            obj.L(numLayers) = BackPropLayer(inputs,outputs,transferFunction);
            
        end
        
        function obj = calcOutput(obj,input)
        nextLayer = input;
        for i = 1:length(obj.L)
        obj.L(i) = obj.L(i).forward(nextLayer);
        nextLayer = obj.L(i).out;
        end
        display(nextLayer);
        end
        
        function obj = doBackprop(obj,learningRate,target)
        %calculate sensitivities 
        obj.L(length(obj.L)) = obj.L(length(obj.L)).firstSensitivity(target);
        for i = (length(obj.L) - 1):-1:1
        obj.L(i) = obj.L(i).Sensitivity(obj.L(i+1).s,obj.L(i+1).weight);    
        end
        %Calculate new weights and biases
        for i = length(obj.L):-1:2
        obj.L(i) = obj.L(i).newWeight(learningRate,obj.L(i -1).out); 
        obj.L(i) = obj.L(i).newBias(learningRate);
        end
        obj.L(1) = obj.L(1).newWeight(learningRate,1); 
        obj.L(1) = obj.L(1).newBias(learningRate);
        

        %set new weights and biases
        for i = length(obj.L):-1:1
        obj.L(i) = obj.L(i).setWeightBias(obj.L(i).w2,obj.L(i).b2);
        end
        end
        
        function E = meanSquareError(obj,target)
        a = obj.L(length(obj.L)).out;
        E = (target - a)' * (target - a);
        end
    end
end

