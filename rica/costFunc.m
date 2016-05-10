function [cost] = costFunc(W, x, params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
lambda = params.lambda;
epsilon = .01;

activation1 = W * x;
activation3 = (W' * activation1) - x;
activation4 = activation3.^2;

%cost function
recCost = sum(sum(activation4)) / 2;
sparseCost = sqrt(activation1.^2 + epsilon);
sparseCost = lambda * norm(sparseCost, 1);
cost = recCost + sparseCost;
end

