function [Z, V] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
display('preparing to find eigenvalues')
sigma = x * x';

sigma = sigma ./ size(x, 2);

display('generating eigenvalues')
%find eigenvectors and values
[U, D] = eig(sigma);
D = fliplr(sum(D));
U = fliplr(U);

xRot = ones(size(U, 2), size(x, 2));
for i = 1:size(x, 3)
    xRot(:, i) = U' * x(:, i);
end

xRot = U' * x;

display('final num eigenvalues: ')
size(D, 2)

xPCAWhite = ones(size(xRot));
regularizer = 1 ./ sqrt(D + epsilon);


regularizer = repmat(regularizer', 1, size(xPCAWhite, 2));
xPCAWhite = regularizer .* xRot;

xZCAWhite = U * xPCAWhite;
Z = xZCAWhite;

%%%%----------------------

regularizer = 1 ./ sqrt(D + epsilon);
regularizer = repmat(regularizer, size(U, 2), 1);
V = regularizer .* U * U';
Z = V * x;
end