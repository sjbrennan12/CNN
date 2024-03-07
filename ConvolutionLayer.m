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

        p %pool layer
        
        resizedP

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
            output(:,:,i) = conv2(obj.Kernels(:,:,1,i),input(:,:,1),'full');   
            for j= 1:obj.KernelDepth
            output(:,:,i) = output(:,:,i) + conv2(obj.Kernels(:,:,j,i),input(:,:,j),'full');
            end
            output(:,:,i) = output(:,:,i) + obj.bias(i);
            end

            temp = [];
            obj.p = output;
            %pooling 
            for l = 1:obj.Outsize
            [rows, cols] = size(output(:,:,l));
            if(mod(cols,2) ~= 0)
            output(:,rows + 1,l) = zeros(rows,1);
            [rows, cols] = size(output(:,:,l));
            end
            if(mod(rows,2) ~= 0)
            output(rows + 1, :,l) = zeros(1,cols);
            end
            obj.resizedP = output;
            
            temp(:,:,l) = obj.pooling(output(:,:,l));
            end

            obj.n = temp;%before transfer function
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

       

        function [obj] = Sensitivity(obj,s2,k2)
            sensitivity = [];
            for i = 1:length(k2(:,:,1,:))
               for j = 1:length(k2(:,:,:,1))
            sensitivity(:,:,j,i) = derrelu(obj,obj.n(:,:,j))* conv2(s2(:,:,i),rot90(rot90(k2(:,:,j,i))),'valid');
               end
           end
           
         beforePool = [];
        [rows, cols] = size(sensitivity(:,:,1));
            for l = 1:obj.Outsize
                beforePool(:,:,l) = zeros(size(obj.resizedP(:,:,1)));
             
                for x = 1:rows
                    for y = 1:cols
                        for i = ((x-1)*obj.Pool + 1):((x-1)*obj.Pool + obj.Pool)
                        for j = ((y-1)*obj.Pool + 1):((y-1)*obj.Pool + obj.Pool)  
                        if(obj.n(x,y,l) == obj.resizedP(i,j,l))
                           beforePool(i,j,l) =  sensitivity(x,y,l);    
                        end
                        end
                        end
                    end
                end
            end
        obj.s = beforePool(1:length(obj.p(1,:,1)),1:length(obj.p(:,1,1)),:);


        end
        
         function output = flattenOutput(obj,input)
        %3d input from convolutional neural network to vector
        num = 1;
        output = [];
        [row,col,depth] = size(input);
        for i = 1:depth
            for j = 1:row
                for k = 1:col
                output(num) = input(j,k,i);
                num = num +1;
                end
            end
        end
         end

        function output = convertOutputSize(~,input,template)
        num = 1;
        output = [];
        [row,col,depth] = size(template);
        for i = 1:depth
            for j = 1:row
                for k = 1:col
                output(j,k,i) = input(num);
                num = num +1;
                end
            end
        end
        end

        function obj = vectorSensitivity(obj,s2,w2)%calculate snesitivity for first convolutional layer before fully connected
        f = obj.derrelu(obj.n);
        f = obj.flattenOutput(f)';
        sensitivity = f.*w2'*s2;
        sensitivity = obj.convertOutputSize(sensitivity,obj.out);
        beforePool = [];
        [rows, cols] = size(sensitivity(:,:,1));
            for l = 1:obj.Outsize
                beforePool(:,:,l) = zeros(size(obj.resizedP(:,:,1)));
             
                for x = 1:rows
                    for y = 1:cols
                        for i = ((x-1)*obj.Pool + 1):((x-1)*obj.Pool + obj.Pool)
                        for j = ((y-1)*obj.Pool + 1):((y-1)*obj.Pool + obj.Pool)  
                        if(obj.n(x,y,l) == obj.resizedP(i,j,l))
                           beforePool(i,j,l) =  sensitivity(x,y,l);    
                        end
                        end
                        end
                    end
                end
            end
        obj.s = beforePool(1:length(obj.p(1,:,1)),1:length(obj.p(:,1,1)),:);

        end


        function obj = newWeight(obj,learningRate, prevA)
            % Update kernel
            obj.Kernel2 = obj.Kernels - learningRate * calcGradient(obj,prevA);
        end

        function grad = calcGradient(obj,prevA)
            %provides the gradient for functions in the Network class
            grad = [];
            for i=1:length(prevA(1,1,:))
            %grad(:,:,i) = zeros(length(obj.Kernels(:,1,1,1)),length(obj.Kernels(1,:,1,1)));
            for j=1:length(obj.s(1,1,:))
            grad(:,:,i,j) = conv2(obj.s(:,:,j),prevA(:,:,i),"valid");
            end
            end

            for i = 1:length(grad(1,1,1,:))
            for j = 1:length(grad(1,1,:,1))
            obj.Kernel2(:,:,j,i) = obj.Kernels(:,:,j,i) + grad(:,:,j,i);
            end
            end
        end
        function obj = newBatchWeight(obj,learningRate,q)
          
        end
        
        function obj = newBatchBias(obj,learningRate,q)
          
        end

        function obj = newBias(obj,learningRate)
        % update bias
        for i=1:obj.Outsize
        total = sum(obj.s(:,:,i),'all');
        total = total / length(obj.s(:,:,1));
        obj.b2(i) = obj.bias(i) - learningRate*total;
        end

        end
        function[obj] = setWeightBias(obj,w,b)
          % set weights and biases with given weights and bias matrices
           obj.Kernels = w;
           obj.bias = b;  
        end
    end


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

