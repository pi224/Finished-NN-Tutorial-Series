function [y] = Sigmoid(x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x = 1 + exp(x);
y = 1 ./ x;
end

