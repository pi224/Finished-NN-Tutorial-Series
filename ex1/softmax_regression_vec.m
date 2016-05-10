function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);
  
  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2) + 1;
  
  % initialize objective value and gradient.

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
%theta = 784 x 10 matrix
theta(:, num_classes) = 0;
activations = theta' * X;

oneFunction = (1:num_classes)';
oneFunction = repmat(oneFunction, 1, size(X, 2));
y = repmat(y, num_classes, 1);
oneFunction = oneFunction == y;

P = exp(activations);
P = P ./ repmat(sum(P, 1), num_classes, 1);
f = oneFunction .* log(P);
f = -sum(sum(f));

g = X * (oneFunction - P)';
g = -g(:, 1:num_classes - 1);

g=g(:); % make gradient a vector for minFunc

