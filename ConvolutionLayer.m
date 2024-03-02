classdef ConvolutionLayer
     properties
        Kernels % store kernel matrixs in 4d matrix 
        
        Pool %store pooling matrix size

        bias
       
        Outsize%depth of output matrix
        
        KernelDepth%depth of kernel

        transfer_function % RELU or other transfer functions
        

        n %input into transfer function
        

        in %input into layer

        out %output of layer

        s %sensitivity vector

        Kernel2 %new kernel  

        b2 %new bias


    end

    methods
        function obj = ConvolutionLayer(OutputSize, KernelWidth,KernelHeight, KernelDepth,poolSize, transfer_function)
           
           
                for i = 1:OutputSize%output layer depth
                obj.bias(i) = (-0.5 + 1 * rand());%bias is applied to every output in matrix
                for j = 1:KernelDepth%number of kernels
                obj.Kernels(:,:,j,i) = (-0.5 + 1 * rand(KernelWidth, KernelHeight))';
                end
                end
                obj.Pool = poolSize;
                obj.Outsize = OutputSize;
                obj.KernelDepth = KernelDepth;

            % Assign the transfer_function string to the transfer_function property.
            obj.transfer_function = transfer_function;
        end

        function [obj] = forward(obj, input)
           
            obj.in = input;
            for i = 1:obj.Outsize 
            output(:,:,i) = conv2(obj.Kernels(:,:,1,i),input(:,:,1),'same');   
            for j= 1:obj.KernelDepth
            output(:,:,i) = output(:,:,i) + conv2(obj.Kernels(:,:,j,i),input(:,:,j),'same');
            end
            output(:,:,i) = output(:,:,i) + obj.bias(i);
            end

            temp = [];
            %pooling 
            for l = 1:obj.Outsize
            temp(:,:,l) = obj.pooling(output(:,:,l));
            end

            obj.n = output;%before transfer function
            output = temp;
            switch(obj.transfer_function)% transfer functions
                case('relu')
                    obj.out = obj.relu_(output);
                case('logsig')
                   %obj.out = obj.logsig_(output)';% return output after transfer function
                case('purelin')
                   %obj.out = obj.purelin_(output)';% return output after transfer function
            end
             
          
        end

        function pooled = pooling(obj,input)
            [rows, cols] = size(input);
            pooled = zeros(round(rows/obj.Pool),round(cols/obj.Pool));
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
        end

       

        function [obj] = Sensitivity(obj,s2,w2)
           
        end

        function obj = newWeight(obj,learningRate, prevA)
            
        end

        function grad = calcGradient(obj,prevA)
            %provides the gradient for functions in the Network class
            grad = obj.s.*prevA';
        end
        function obj = newBatchWeight(obj,learningRate,q)
          
        end
        
        function obj = newBatchBias(obj,learningRate,q)
          
        end

        function obj = newBias(obj,learningRate)
            
        end
        function[obj] = setWeightBias(obj,w,b)
           
        end
    end


    methods (Access = private)
        %Todo Implement other transfer functions besides relu
        % Private methods to be used within the BackProp Layer
        function a = relu_(~,p)
        a = max(0,p);
        end

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

