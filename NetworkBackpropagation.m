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
        %display(nextLayer);
        end

        function obj = doBatchBackprop(obj,learningRate,inputs,targets)

        [row,col] = size(inputs);
         obj = obj.calcOutput(inputs(:,1));
         obj.L(length(obj.L)) = obj.L(length(obj.L)).firstSensitivity(targets(:,1));
         for i = (length(obj.L) - 1):-1:1
         obj.L(i) = obj.L(i).Sensitivity(obj.L(i+1).s,obj.L(i+1).weight);    
         end

         for i = length(obj.L):-1:2
         obj.L(i).AS = obj.L(i).calcGradient(obj.L(i-1).out); 
         obj.L(i).allS = obj.L(i).s;
         end
         obj.L(1).AS = obj.L(1).calcGradient(obj.L(1).in); 
         obj.L(1).allS = obj.L(1).s;

        for p = 2:col
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

        for i = 1:length(obj.L)
        obj.L(i) = obj.L(i).newBatchWeight(learningRate,col); 
        obj.L(i) = obj.L(i).newBatchBias(learningRate,col);
        end
        

       
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
        obj.L(1) = obj.L(1).newWeight(learningRate,obj.L(1).in); 
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

