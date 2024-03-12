classdef BackPropLayer
     %BACKPROPLAYER Implements a network layer that calculates layer outputs, 
     % sensitivities, and parameters of a fully connected layer. 
     % Methods:
     %    - BackPropLayer (Constructor)
     %    - forward
     %    - print
     %    - firstSensitivity
     %    - Sensitivity
     %    - newWeight
     %    - calcGradient
     %    - newBatchWeight
     %    - newBatchBias
     %    - newBias
     %    - setWeightBias
     % Transfer functions:
     %    - logisig
     %    - derlogsig
     %    - purelin
     %    - derpurelin
     
     properties
        weight
        % Weight Matrix to matrix multiply with the input.
        % is either specified by the client or generated using random
        % values uniformly distributed in the range [-1, +1]
        %
        % When generated, the dimensions of the matrix will be 
        % arg1 = rows = number of inputs
        % arg2 = cols = number of outputs

        bias
        % Bias Vector to add to each neuron/row of input * weight matrix.
        % Variable is either specified by the client or generated using
        % random values uniformly distributed in the range [-1, +1]
        %
        % When generated, the vector will be
        % length = arg2 = number of outputs

        transfer_function
        % Transfer Function is a string that specifies if a Hard Limit("harmlim") or 
        % Symmetrical Hard Limit("harlims") private transfer function will be used.

        n % net input of the layer

        in % layer input

        out % layer output after the transfer function

        s % layer sensitivities 

        w2 % new calculated weight value

        b2 % new calculated bias value

        AS = []; % all outputs calculated for a batch of inputs

        allS = []; % all sensitivities calculated for a batch of inputs

    end

    methods
        function obj = BackPropLayer(arg1, arg2, transfer_function)
            %BACKPROPLAYER constructs an instance of this class
            %   @param arg1 Represents the number of inputs the layer will take,
            %               or the weight matrix
            %   @param arg2 Represents the number of outputs the layer will
            %               output, or the bias vector  
            %   @param transfer_function specifies the type of transfer
            %                            function to use in the 
            %                            layer.            

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

            % Assign the transfer_function
            obj.transfer_function = transfer_function;
        end % end of constructor

        function [obj] = forward(obj, input)
            %FORWARD takes the object and input vector and produces the output.
            %   @param input the input vector to the layer, of size (Rx1) where
            %                R is the number of inputs to the network. 
            %   uses function a = Wp + B, where a is the output vector, W is the
            %   weight matrix, p is the input vector and B is the bias vector, 
            %   to calculate the output of the layer
           
            obj.in = input;
            output = obj.bias + (obj.weight*input); % wp+b
            obj.n = output; % before transfer function

            switch(obj.transfer_function)% transfer functions
                case('logsig')
                   obj.out = obj.logsig_(output')'; % output with logsig
                case('purelin')
                   obj.out = obj.purelin_(output')'; % output with purelin
            end
        end % end of forward function

        function print(obj) 
            %PRINT Prints weights and biases to console
            disp("Weights");
            disp(obj.weight);
            disp("Biases");
            disp(obj.bias);
        end % end of print function

        function [obj] = firstSensitivity(obj,t)
            %FIRSTSENSITIVITY calculates the sensativity for the final layer
            %   @param t the expected output of the network
            %   calculates sensitivity of layer with formula: s = -2dF(n)(t-a)

            switch(obj.transfer_function)% transfer functions
                case('logsig')
                    obj.s = -2*obj.derlogsig(obj.out)*(t-obj.out);
                case('purelin')
                    obj.s = -2*obj.derpurelin(obj.out)*(t-obj.out);
            end
        end % end of firstSensitivity function

        function [obj] = Sensitivity(obj,s2,w2)
            %SENSITIVITY calculates the sensativity of layers (not final)
            %   @param s2 sensitivity matrix of next layer
            %   @param w2 weight matrix of next layer
            %   calculates sensitivity of layer with formula: s = dF(n)Ws

            switch(obj.transfer_function)% transfer functions
                case('logsig')
                    obj.s = obj.derlogsig(obj.out) * (w2' * s2);
                case('purelin')
                    obj.s = obj.derpurelin(obj.out) * (w2' * s2);
            end
        end % end of Sensitivity function

        function obj = newWeight(obj,learningRate, prevA)
            %NEWWEIGHT updates new weights for layer.
            %   @param learning rate stable learning rate
            %   @param prevA previous layer output

            % new W = old W - alpha * sensititivities * prevA'
            obj.w2 = obj.weight - (learningRate*obj.s.*prevA');
        end

        function grad = calcGradient(obj,prevA)
            %CALCGRADIENT calculates the gradient of the layer
            %   @param prevA layer output

            grad = obj.s.*prevA';
        end

        function obj = newBatchWeight(obj,learningRate,q)
            %NEWBATCHWEIGHT updates new weights for batch update
            %   @param q batch size
            
            obj.weight = obj.weight - ((learningRate /q)*obj.AS);
        end
        
        function obj = newBatchBias(obj,learningRate,q)
            %NEWBATCHWEIGHT updates new biases for batch update
            %   @param q batch size
            obj.bias =  obj.bias -((learningRate /q).*obj.allS);
        end

        function obj = newBias(obj,learningRate)
            %NEWBIAS updates new biases for layer.
            %   @param learning rate stable learning rate
            %   @param prevA previous layer output

            % new B = old B - alpha * sensitivities
            obj.b2 = obj.bias -( learningRate.*obj.s);
        end

        function[obj] = setWeightBias(obj,w,b)
            %SETWEIGHTBIAS used to update the weights and biases 
            %   @param w new weights to be set
            %   @param b new biases to be set

            obj.weight = w;
            obj.bias = b;
        end
    end % end of public methods


    methods (Access = private)
        % Private methods to be used within the BackProp Layer

        function a = derlogsig(~,p)
            %DERLOGSIG returns dervivitive of logsig
            a = (eye(length(p)).*((1-p).*p));
            %Verified with example sensitivity calculations 
        end

        function a = derpurelin(~,p)
            %DERPURELIN dervititive of purelin
            a = 1; 
        end
        
        function a = logsig_(~,p)
            %LOGSIG_ sigmoid transfer function, equation: 1/(1+e^-x)
            a = logsig(p);
        end

        function a = purelin_(~, p)
            %PURELIN_ pu
            %   Takes an input p and outputs 1 if the value positive or
            %   outputs 0 if the value is negative.
            %   a = -1, n < 0
            %   a = +1, n => 0

            a = p;
        end
    end % end of private methods
end % end of BackpropLayer class

