%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
%ICA code
lambda = params.lambda;
epsilon = .01;

activation1 = W * x;
activation3 = (W' * activation1) - x;
activation4 = activation3.^2;

%cost function
recCost = sum(sum(activation4)) / 2;
sparseCost = sum(sum(sqrt(activation1.^2 + epsilon)));
cost = recCost + lambda * sparseCost;

%reconstruction gradient
recGrad = W * activation3 * x';
recGrad = recGrad + activation1 * activation3';

%sparsity gradient
sparseGrad = activation1 ./ sqrt(activation1.^2 + epsilon);
sparseGrad = sparseGrad * x';
sparseGrad = lambda .* sparseGrad;

Wgrad = recGrad + sparseGrad;

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);

