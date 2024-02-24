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

        in %layer inputs 

        out % layer outputs after the transfer function

        s % layer sensitivities 

        w2 % new calculated weight value

        b2 % new calculated bias value

        AS = []; % all outputs calculated for a batch of inputs

        allS = []; % all sensativities calculated for a batch of inputs

    end

    methods
        function obj = BackPropLayer(arg1, arg2, transfer_function)
            %BackPropLayer constructs an instance of this class
            %   @param arg1  
            %               Represents
            %               the number of inputs the layer will
            %               take, or the weight matrix
            %   @param arg2 Eepresents
            %               the number of outputs the layer will
            %               output, or the bias vector  
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
            %calculates the sensativity for the layer at the end of the
            %network using the equation s = -2dF(n)(t-a)
        switch(obj.transfer_function)% transfer functions
        case('logsig')
            obj.s = -2*obj.derlogsig(obj.out)*(t-obj.out);
        case('purelin')
            obj.s = -2*obj.derpurelin(obj.out)*(t-obj.out);
        end
        end

        function [obj] = Sensitivity(obj,s2,w2)
            %calculates the sensativity for all layers except the final
            %layer by using backpropagation of the previous calculated
            %sensativities and previous layers weight. s = dF(n)Ws
        switch(obj.transfer_function)% transfer functions
        case('logsig')
            obj.s = obj.derlogsig(obj.out) * (w2' * s2);
        case('purelin')
            obj.s = obj.derpurelin(obj.out) * (w2' * s2);
        end
        end

        function obj = newWeight(obj,learningRate, prevA)
            %calculates the updated weight by using the equation 
            % new W = old W - as *prevA
            obj.w2 = obj.weight - (learningRate*obj.s.*prevA');
        end

        function grad = calcGradient(obj,prevA)
            %provides the gradient for functions in the Network class
            grad = obj.s.*prevA';
        end
        function obj = newBatchWeight(obj,learningRate,q)
            %batch propagation version of the weight calculation
            obj.weight = obj.weight - ((learningRate /q)*obj.AS);
        end
        
        function obj = newBatchBias(obj,learningRate,q)
            %batch propagation version of the bias calculation
            obj.bias =  obj.bias -((learningRate /q).*obj.allS);
        end

        function obj = newBias(obj,learningRate)
            %bias calculation that uses the equation new b = b - as
            obj.b2 = obj.bias -( learningRate.*obj.s);
        end
        function[obj] = setWeightBias(obj,w,b)
            %used to update the weights and biases to the inputs and this
            %allows for initial weights to be set and the new biases and
            %weights to be set
            obj.weight = w;
            obj.bias = b;
        end
    end


    methods (Access = private)
        % Private methods to be used within the BackProp Layer

        function a = derlogsig(~,p)
        a = (eye(length(p)).*((1-p).*p));%returns dervivitive of logsig for calulating the sensativity
        %Verified with example sensitivity calculations 
        end

        function a = derpurelin(~,p)
        a = 1;% dervititive of purelin 
        end
        
        function a = logsig_(~,p)
            %equation 1/(1+e^-x)
        a = logsig(p);%matlab provided function
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

