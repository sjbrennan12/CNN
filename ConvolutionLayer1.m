classdef ConvolutionLayer1 < handle
    %CONVOLUTIONLAYER1 CONVOLUTIONLAYER implements a netowrk layer that calculates the
     %convoliution, activation, and pooling
     % Methods:
     %    - ConvolutionLayer1 (constructor)
     %    - forward
     %    - maxPool
     %    - firstSensitivity
     %    - uppool
     %    - Sensitivity
     %    - print
     %    - calcGradient
     %    - newBias
     %    - setWeightBias
     %    - newWeight
     % transfer functions:
     %    - (relu)
     %    - derrelu
    
    properties
        Kernels 
        % stores the kernel matricies used for convolution
        % it is a 4d matrix, fxfxcxo where fxf is the input dimentions
        % (kernel width x kernel height), c is the number of input
        % channels (kernel depth), and o is the number of output channels
        % (number of filters)

        poolSize %size of pool 

        b % bias vector applied to each output channel
       
        numOutputs % depth of output matrix, num of out channels, num of kernels
        
        kernelDepth % depth of kernel, number of input channels

        kernelSize % height and width of kernel

        tf % RELU or other transfer functions
        
        n % net input of layer: input into transfer function
        
        in % input into layer, 3D matrix (h x w x num_in_channels)

        out % output of layer 3D matrix (h x w x num_out_channels)

        convOut % output after convolution + bias

        s % sensitivity 

        p % pool layer: result after pooling

        cood % pool coordinates of max pool 

        Kernel2 % new kernel  

        b2 % new bias
    end
    
    methods
        function obj = ConvolutionLayer1(OutputSize, KernelSize, KernelDepth,poolSize, tf)
            %CONVOLUTIONLAYER Initializes the layer
            %   @param OutputSize number of output channels
            %   @param KernelSize height and width of kernel
            %   @param KernelDepth depth of kernel, number of input channels
            %   @param poolSize pool height and width
            %   @param tf transfer function of layer
            
            obj.poolSize = poolSize;
            obj.numOutputs = OutputSize;
            obj.kernelDepth = KernelDepth;
            obj.tf = tf;
            obj.kernelSize = KernelSize;

            obj.b = -0.5 + 1 * rand(1,OutputSize);
            obj.Kernels = -0.5 + 1 * rand(KernelSize, KernelSize, KernelDepth, OutputSize) ; 
        end
        
        function [obj] = forward(obj,input)
            %FORWARD performs the forward pass through the layer 
            %   @param input network input in format (height x width x channels)
            %   @return obj with calculated output
            %   First performs convolution with bias addition for each output
            %   channel, then performs relu activation function and then maxpool

            [in_h, in_w, in_c] = size(input); % input height, weight, channels
            obj.in = input;
            if(in_c ~= obj.kernelDepth)
                error("Input channels not the same as kernel depth")
            end

            % convolution with kernel of input and add bias to each convolution
            % get dimentions of convolved output
            o_h = (in_h - obj.kernelSize) + 1; % output height
            o_w = (in_w - obj.kernelSize) + 1; % output width
            convOutput =  zeros(o_h,o_w,obj.numOutputs); % store output here
            
            % convolution
            for i=1:obj.numOutputs
                for j=1:in_c
                    convOutput(:,:,i) = convOutput(:,:,i) + ...
                    conv2(input(:,:,j), obj.Kernels(:,:,j), "valid");
                end
                convOutput(:,:,i) = convOutput(:,:,i) + obj.b(i);
            end
            obj.convOut = convOutput;

            % max pool the output for each output channel and store as a 3d
            % matrix
            obj.p = maxPool(obj,convOutput, obj.poolSize);

            obj.n = obj.p; % store net input before activation

            % the final is a 3d matrix 
            % apply relu to 3d matrix and get a 3d matrix answer if possible
            % make switch statement for other tf functions
            obj.out = max(0,obj.n);
           
            % get downsampled output
        end % end of forward function

        function pooledOutput = maxPool(obj,input, poolSize)
            %MAXPOOL performs maxpool on given input
            %   @param input to be downsampled
            %   @param poolSize dimentions of pool matrix of size (size x size)
            %   @return downsampled output

            [in_h, in_w, in_c] = size(input); % input height, weight, channels
            % get dimentions of pool output
            o_h = floor(in_h/poolSize); % output height
            o_w = floor(in_w/poolSize); % output width
            pooledOutput = zeros(o_h, o_w, in_c); % store output here
            obj.cood = zeros(o_h, o_w, in_c, 2);

            % max pooling
            for i=1:in_c
                for h = 1:poolSize:in_h
                    for w = 1:poolSize:in_w
                        end_h = min(h + poolSize -1, in_h);
                        end_w = min(w + poolSize -1, in_w);
                        pool_region = input(h:end_h, w:end_w, i);
                        [maxval, index] = max(pool_region(:));
                        [pool_row, pool_col] = ind2sub(size(pool_region), index);
                        max_row = h - 1 + pool_row;
                        max_col = w - 1 + pool_col;
                        pool_row = ceil(h/poolSize);
                        pool_col = ceil(w/poolSize);
                        pooledOutput(pool_row, pool_col, i) = maxval;
                        obj.cood(pool_row, pool_col, i, :) = [max_row, max_col];

                    end
                end

            end
        end % end of maxPool function

        function obj = firstSensitivity(obj, s2, w2)
            %FIRSTSENSITIVITY calcualte sensitivity from fc layer to conv layer
            %   @param s2 sensitivity of FC layer (next layer)
            %   @param w2 weights of FC layer (next layer)
            
            dF_n = derrelu(obj,obj.n); % size: (height x weight x depth)

            % flatten the vector
            [row, col, depth] = size(dF_n);
            df_n = reshape(permute(dF_n, [2, 1, 3]),[row * col * depth,1]);
            df_n = df_n';

            % propagate sensitivities backward
            jacob_matrix = diag(df_n);
            obj.b2 = jacob_matrix;
            sensitivity = (jacob_matrix * w2') * s2;

            %reshape sensitivities
            [rows, cols, depth] = size(obj.out);
            total_elements = rows * cols * depth;
            if length(sensitivity) ~= total_elements
                error("incorrenct dimentions of new sensitivity matrix")
            end
            reshaped_matrix = reshape(sensitivity, cols, rows, depth);
            reshaped_matrix = permute(reshaped_matrix, [2, 1, 3]);

            % uppool sensitivities
            sensitivity_matrix = uppool(obj, reshaped_matrix);
            obj.s = sensitivity_matrix;

            % sens = padSensitivity(obj, sensitivity_matrix);
            % obj.Kernel2 = sens;


            
           %continue by taking the pool and recreating the original input 3d matrix with the coordinates
           %the coordinates get the sensitivity everything else is 0
        end % end of firstSensitivity function

        function original = uppool(obj, matrix) 
            % [pool_h,pool_w,pool_c, ~] = size(obj.cood);
            [matrix_h, matrix_w, matrix_c] = size(matrix);

            original = zeros(size(obj.convOut));
            for c = 1:matrix_c
                for h = 1:matrix_h
                    for w =1:matrix_w
                        % get corresponding pool coordinates for the current
                        % element
                        max_row = obj.cood(h,w,c,1);
                        max_col = obj.cood(h,w,c,2);

                        original(max_row, max_col,c) = matrix(h,w,c);
                    end
                end
            end
        end % end of function uppool

        function obj = Sensitivity(obj, s2, k2)
            % UPDATESENSITIVITY updates the sensitivity from the next convolutional layer to the current convolutional layer
            %   @param s2 sensitivity matrix of the next convolutional layer
            %   @param k2 kernels of the next convolutional layer
            %   @return sensitivity sensitivity matrix of the current convolutional layer
        
            sensitivity = zeros(size(obj.out)); % Initialize sensitivity matrix
            % Loop through each output channel of the next layer
            for i = 1:obj.numOutputs
                % Loop through each input channel of the next layer
                for j = 1:obj.kernelDepth
                    % Compute derivative of activation function (ReLU)
                    dF_n = derrelu(obj, obj.n(:,:,i));
                    
                    % Rotate kernel for convolution
                    rotatek_kernel = rot90(rot90(k2(:,:,j,i)));
                    
                    % Perform convolution operation
                    rot_conv = conv2(s2(:,:,i), rotatek_kernel, 'valid');
                    disp(size(dF_n));
                    disp(size(rot_conv));

                    % Reshape rot_conv to match the size of dF_n
                    rot_conv_resized = imresize(rot_conv, size(dF_n));
                    % Compute sensitivity contribution
                    sensitivity(:,:,j,i) = dF_n .* rot_conv_resized; % Element-wise multiplication
                    
                    % Uncomment the following line if you want to visualize the sizes for debugging
                    % disp(size(sensitivity(:,:,j,i)));
                end
            end
            
            % Further processing if needed
            
            % Update sensitivity property
            obj.s = sensitivity;

        end
        

        function print(obj)
            % PRINT prints the layer  dimentions
            disp("Kernel: ")
            disp(obj.Kernels)
            disp("bias: ")
            disp(obj.b)
        end % end of print function

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
        end % end of calcGradeint function

        function obj = newBias(obj,learningRate)
            % update bias
            for i=1:obj.numOutputs
            total = sum(obj.s(:,:,i),'all');
            total = total / length(obj.s(:,:,1));
            obj.b2(i) = obj.b(i) - learningRate*total;
            end
        end


        function[obj] = setWeightBias(obj,w,b)
            % set weights and biases with given weights and bias matrices
            obj.Kernels = w;
            obj.b = b;  
        end

        function obj = newWeight(obj,learningRate, prevA)
            % Update kernel
            obj.Kernel2 = obj.Kernels - learningRate * calcGradient(obj,prevA);
        end
    end % end of public methods
    methods (Access = private)
        function a = derrelu(~,p)
            a = p >= 0;
        end
            
    end % end of private methods
end

