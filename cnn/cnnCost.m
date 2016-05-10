function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
for imageIndex = 1:numImages
    for filterIndex = 1:numFilters
        filter = Wc(:, :, filterIndex);
        bias = bc(filterIndex);
        filter = rot90(filter, 2);
        
        image = images(:, :, imageIndex);
        
        activation = conv2(image, filter, 'valid'); %finding activation
        activation = activation + bias;
        activation = Sigmoid(activation);
        
        activations(:, :, filterIndex, imageIndex) = activation;
        
        %pooling using mean
        onesMatrix = ones(poolDim);
        pooledActivation = conv2(activation, onesMatrix, 'valid');
        [rows, cols] = size(pooledActivation);
        
        pooledActivation = pooledActivation(1:poolDim:rows, 1:poolDim:cols);
        pooledActivation = pooledActivation ./ poolDim^2;
        
        activationsPooled(:, :, filterIndex, imageIndex) = pooledActivation;
    end
end

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.

%%% YOUR CODE HERE %%%
[numNodes, numImages] = size(activationsPooled);

preds = Wd * activationsPooled;
non_sigmoid_act = preds;
preds = preds + repmat(bd, 1, size(activationsPooled, 2));

probs = preds;

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%

cost_per_image = preds .* 0;

for i = 1:numImages
    a = non_sigmoid_act(:, i);
    label = labels(i);
    normalizer = sum(exp(a));
    
    P = exp(a)/normalizer;
    
    oneFunction = (1:10) == label;
    cost_per_image(:, i) = -(oneFunction' - P);
    cost = cost - sum(oneFunction' .* log(P));
end

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%

%computing gradient for softmax layer
soft_weightGrad = zeros(size(Wd));
soft_biasGrad = zeros(size(bd));
conv_weightGrad = Wc .* 0;
conv_biasGrad = bc .* 0;

%something, somewhere is causing the gradients to be incorrect whenever the
%correct gradient is supposed to be 0 FIND IT!!!!!!
for i = 1:numImages
    currentCost = cost_per_image(:, i);
    X = activationsPooled(:, i);
    delta = currentCost;
    
    wGrad = delta * X';
    
    soft_weightGrad = soft_weightGrad + wGrad;
    
    %calcualte convolved gradients
    delta = Wd' * delta;
    dim = size(activations, 1) / poolDim;
    delta = reshape(delta, [dim, dim, numFilters]); %we must reshape delta so that it can be used properly
    
    for filterIndex = 1:numFilters
        activation = activations(:, :, filterIndex, i);
        delta_conv = delta(:, :, filterIndex);
        onesMatrix = ones(poolDim);
        
        delta_conv = kron(delta_conv, onesMatrix) ./ poolDim^2 .* activation .* (1 - activation);
        image = images(:, :, i);
        
        delta_conv = rot90(delta_conv, 2);
        
        wGrad = conv2(image, delta_conv, 'valid');
        bGrad = sum(sum(delta_conv));
        
        conv_weightGrad(:, :, filterIndex) = conv_weightGrad(:, :, filterIndex) + wGrad;
        conv_biasGrad(filterIndex) = conv_biasGrad(filterIndex) + bGrad;
    end
end

Wd_grad = Wd_grad + soft_weightGrad;
bd_grad = bd_grad + soft_biasGrad;
Wc_grad = Wc_grad + conv_weightGrad;
bc_grad = bc_grad + conv_biasGrad;

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
