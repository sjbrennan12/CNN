classdef BackPropLayer
     properties
        weight
        %Weight Matrix to matrix multiply with the input.
        %   is either specified by the client or generated using using random
        %   values uniformly distributed in the range [-1, +1]
        %
        %   When generated, the dimensions of the matrix will be 
        %   arg1 = rows = number of inputs
        %   arg2 = cols = number of outputs

        bias
        %Bias Vector to add to each neuron/row of input * weight matrix.
        %   Variable is either specified by the client or generated using
        %   random values uniformly distributed in the range [-1, +1]
        %
        %   When generated, the vector will be
        %   length = arg2 = number of outputs

        transfer_function
        %Transfer Function is a string that specifies if a Hard Limit("harmlim") or 
        %   Symmetrical Hard Limit("harlims") private transfer function will be used.

        n
        %holds the result before the forward function to allow
        %backprop

        in

        out

        s

        w2

        b2

        AS = [];

        allS = [];

    end

    methods
        function obj = BackPropLayer(arg1, arg2, transfer_function)
            %BackPropLayer constructs an instance of this class
            %   @param arg1 Can be represented in two ways. 
            %               If arg is a scalar, then it represents
            %               the number of inputs the perceptron layer will
            %               take, and the number of rows for the weight
            %               matrix.
            %               If arg is a matrix then assign it to weight
            %               property.
            %   @param arg2 Can be represented in two ways.
            %               If arg is a scalar, then is represents
            %               the number of outputs the perceptron layer will
            %               output, the number of columns for the weight 
            %               matrix, and the length of the bias vector. 
            %               If the argument is a vector then it represents
            %               the bias vector to use for the perceptron
            %               layer.
            %   @param transfer_function specifies the type of transfer
            %                            function to use in the 
            %                            layer.
            %                            

            if size(arg1) == 1
                % If arg1 and arg2 are scalars, then generate a weight
                % matrix and bias vector where the values are uniformly
                % distributed in the range [-1, +1].

                obj.weight = (-1 + 2 * rand(arg1, arg2))';
                obj.bias = (-1 + 2 * rand(1, arg2))';

            else
                % If arg1 and arg2 are a matrix and vector, then assign
                % them to the weight and bias properties.

                obj.weight = arg1;
                obj.bias = arg2;
            end

            % Assign the transfer_function string to the transfer_function property.
            obj.transfer_function = transfer_function;
        end

        function [obj] = forward(obj, input)
           %forward takes the object and input vector and produces the
            %output from the layer after the transfer function
            %performs multiplication between the weight matrix and input vector and then adds the bias vector to 
            %the result before performing the transfer function
           
            obj.in = input;
            output = obj.bias + (obj.weight*input);
            obj.n = output;%before transfer function
            switch(obj.transfer_function)% transfer functions
                case('logsig')
                   obj.out = obj.logsig_(output')';% return output after transfer function
                case('purelin')
                   obj.out = obj.purelin_(output')';% return output after transfer function
            end
             
          
        end

        

        function print(obj) 
            %Prints weights and biases to console
            disp("Weights");
            disp(obj.weight);
            disp("Biases");
            disp(obj.bias);
        end

        function [obj] = firstSensitivity(obj,t)
        switch(obj.transfer_function)% transfer functions
        case('logsig')
            obj.s = -2*obj.derlogsig(obj.out)*(t-obj.out);
        case('purelin')
            obj.s = -2*obj.derpurelin(obj.out)*(t-obj.out);
        end
        end

        function [obj] = Sensitivity(obj,s2,w2)
        switch(obj.transfer_function)% transfer functions
        case('logsig')
            obj.s = obj.derlogsig(obj.out) * (w2' * s2);
        case('purelin')
            obj.s = obj.derpurelin(obj.out) * (w2' * s2);
        end
        end

        function obj = newWeight(obj,learningRate, prevA, y)
            obj.w2 = y .* obj.weight - ((1-y) * learningRate*obj.s.*prevA');
        end

        function grad = calcGradient(obj,prevA)
            grad = obj.s.*prevA';
        end
        function obj = newBatchWeight(obj,learningRate,q)
            obj.weight = obj.weight - ((learningRate /q)*obj.AS);
        end
        
        function obj = newBatchBias(obj,learningRate,q)
            obj.bias =  obj.bias -((learningRate /q).*obj.allS);
        end

        function obj = newBias(obj,learningRate, y)
            obj.b2 = y .* obj.bias -((1-y) * learningRate.*obj.s);
        end
        function[obj] = setWeightBias(obj,w,b)
            obj.weight = w;
            obj.bias = b;
        end
    end


    methods (Access = private)
        % Private methods to be used within PerceptronLayer

        function a = derlogsig(~,p)
        a = (eye(length(p)).*((1-p).*p));%returns dervivitive of logsig for calulating the sensativity
        %verified with textbook operation 
        end

        function a = derpurelin(~,p)
        a = 1;
        end
        
        function a = logsig_(~,p)
        a = logsig(p);%matlab function
        end

        function a = purelin_(~, p)
            %Symmetrical Hard Limit Transfer Function
            %   Takes an input p and outputs 1 if the value positive or
            %   outputs 0 if the value is negative.
            %   a = -1, n < 0
            %   a = +1, n => 0

             a = p;
        end
    end
end

