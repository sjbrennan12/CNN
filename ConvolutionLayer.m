classdef ConvolutionLayer
     %CONVOLUTIONLAYER implements a netowrk layer that calculates the
     %convoliution, activation, and pooling
     % Methods:
     %    - ConvolutionLayer
     %    - forward
     %    - pooling
     %    - Sensitivity
     % transfer functions
     %    - purelin
     %    - logsig
     %    - relu

     properties
        Kernels 
        % stores the kernel matricies used for convolution
        % it is a 4d matrix, fxfxcxo where fxf is the input dimentions
        % (kernel width x kernel height), c is the number of input
        % channels (kernel depth), and o is the number of output channels
        % (number of filters)
        
        Pool
        % stores size of pooling matrix for downsampling. the size determines
        % the regions over which the pooling will be performed (PoolxPool). 

        bias 
        % Bias applied to output in output channels. Bias is added to 
        % convolution result before applying the activation function
       
        Outsize %depth of output matrix, the number of output channels
        
        KernelDepth %depth of kernel, number of input channels

        transfer_function % RELU or other transfer functions
        
        n %net input of layer: input into transfer function
        
        in %input into layer, 3D matrix (h x w x num_in_channels)

        out %output of layer 3D matrix (h x w x num_out_channels)

        s %sensitivity 

        p % pool layer: result after pooling

        Kernel2 % new kernel  

        b2 %new bias


    end

    methods
        function obj = ConvolutionLayer(OutputSize, KernelWidth,KernelHeight, KernelDepth,poolSize, transfer_function)
            %CONVOLUTIONLAYER Initializes the layer with kernal matricies, bias,
            % pooliing size, tranfer function and no of output channels of the 
            % layer.
           
            % iterate over each output channel and create bias vector and
            % initialize kernels kernel size 
            for i = 1:OutputSize
                %bias is applied to every output in matrix
                obj.bias(i) = (-0.5 + 1 * rand());

                %iterate over number of input channels (kernel depth) and
                % crate random kernal values between 0.5 and -0.5
                for j = 1:KernelDepth
                    obj.Kernels(:,:,j,i) = (-0.5 + 1 * rand(KernelWidth, KernelHeight))';
                end
            end
            obj.Pool = poolSize;
            obj.Outsize = OutputSize;
            obj.KernelDepth = KernelDepth;

            % Assign the transfer_function string to the transfer_function property.
            obj.transfer_function = transfer_function;
        end % end of constructor

        function [obj] = forward(obj, input)
            %FORWARD performs the forward pass through the layer, applying
            % convolution, bias addition, and transfer function. The input to
            % the network should be in format (height x width x channels)

            obj.in = input;
            % iterate over number of output channels
            for i = 1:obj.Outsize 
                output(:,:,i) = conv2(obj.Kernels(:,:,1,i),input(:,:,1),'same');

                % for each input channel
                for j= 1:obj.KernelDepth
                    output(:,:,i) = output(:,:,i) + conv2(obj.Kernels(:,:,j,i),input(:,:,j),'same');
                end
                output(:,:,i) = output(:,:,i) + obj.bias(i);
            end

            %pooling
            temp = []; 
            for l = 1:obj.Outsize
                temp(:,:,l) = obj.pooling(output(:,:,l));
            end
            
            %before transfer function
            obj.n = output;
            output = temp;
            obj.p = temp;
            switch(obj.transfer_function)% transfer functions
                case('relu')
                    obj.out = obj.relu_(output);
                case('logsig')
                   %obj.out = obj.logsig_(output)';% return output after transfer function
                case('purelin')
                   %obj.out = obj.purelin_(output)';% return output after transfer function
            end
        end % end of forward function

        function pooled = pooling(obj,input)
            %POOLING performs pooling operation on input after convolution

            [rows, cols] = size(input);
            pooled = zeros(round(rows/obj.Pool),round(cols/obj.Pool));

            % iterate over rows and cols of input with step size of pool size
            for i = 1:obj.Pool:rows 
                for j = 1:obj.Pool:cols 
                    max = 0;
                    for x = i:(i+obj.Pool -1)
                        for y = j:(j+obj.Pool -1)
                            if(input(x,y) > max)
                                max = input(x,y);
                            end
                        end
                    end
                    pooled(ceil(i/obj.Pool),ceil(j/obj.Pool)) = max;
                end
            end
        end % end of pooling function

       

        function [obj] = Sensitivity(obj,s2,inputKernel)
            % Calculate Sensitivities for layer

            %Part1 Push sensitivities through pooling 
            beforePool = [];
            [rows, cols] = size(s2);
            for l = 1:obj.Outsize
                beforePool(:,:,l) = zeros(size(obj.n(:,:,1))); 
                for x = 1:rows
                    for y = 1:cols
                        for i = ((x-1)*obj.Pool + 1):((x-1)*obj.Pool + obj.Pool)
                            for j = ((y-1)*obj.Pool + 1):((y-1)*obj.Pool + obj.Pool)  
                                if(obj.n(i,j,l) == obj.p(x,y,l))
                                    beforePool(i,j,l) =  obj.p(x,y,l);    
                                end
                            end
                        end
                    end
                end
            end

           %Part2 Calculate sensitivities before RELU
           %Reverse convolution 
           for i = 1:obj.Outsize
               for j = 1:obj.KernelDepth
                   obj.s(:,:,j,i) = derrelu(obj,beforePool(:,:,i))* ...
                   conv2(s2,rot90(rot90(inputKernel(:,:,j,i))),'full');
               end
           end
        end % end of Sensitivities method

        function obj = newWeight(obj,learningRate, prevA)
            %TODO
        end

        function grad = calcGradient(obj,prevA)
            %provides the gradient for functions in the Network class
            grad = obj.s.*prevA';
        end

        function obj = newBatchWeight(obj,learningRate,q)
          %TODO
        end
        
        function obj = newBatchBias(obj,learningRate,q)
          %TODO
        end

        function obj = newBias(obj,learningRate)
            %TODO
        end

        function[obj] = setWeightBias(obj,w,b)
           %TODO
        end
    end % end of public methods


    methods (Access = private)
        %Todo Implement other transfer functions besides relu
        % Private methods to be used within the BackProp Layer
        function a = relu_(~,p)
            a = max(0,p);
        end
        
        function a = derrelu(~,p)
            a = max(0,p);
            a = min(1,p);
        end

        function a = derlogsig(~,p)
            %returns dervivitive of logsig for calulating the sensativity
            a = (eye(length(p)).*((1-p).*p));
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
    end % end of private methods
end % end of ConvolutionLayer class

