function [y] = Sigmoid(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
y = 1 + exp(-x);
y = 1 ./ y;
end

