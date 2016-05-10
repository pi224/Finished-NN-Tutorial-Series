function [f, g] = softmax_regression_2(theta, X, y)
%same as other softmax regression, except that we optimize over ALL the
%parameters
%reshape theta
n=size(X,1);
theta=reshape(theta, n, []);
num_classes = size(theta,2);

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
g = -g;

g=g(:);
end

