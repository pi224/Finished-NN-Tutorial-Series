function [dy] = Dsigmoid(y)
%Gives derivative of sigmoid activation function
%   with respect to x
dy = y .* (1-y);
end

