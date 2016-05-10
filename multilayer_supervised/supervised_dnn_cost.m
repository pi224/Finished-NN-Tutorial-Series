function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

for i = 1:size(stack) %give stacks something to base recordings off of
    hAct{i} = struct('a', zeros(size(stack{i}.W, 1), 1));
    gradStack{i} = struct('W', zeros(size(stack{i}.W)), 'b', zeros(size(stack{i}.W, 1), 1));
end

%% forward prop
%%% YOUR CODE HERE %%%
numData = size(data, 2);

W1 = stack{1}.W;
b1 = stack{1}.b;
W2 = stack{2}.W;
b2 = stack{2}.b;

output = (W1 * data) + repmat(b1, 1, numData);
output = Sigmoid(output);
hAct{1}.a = output;
non_sigmoid_act = W2 * output;
output = (W2 * output) + repmat(b2, 1, numData);
hAct{2}.a = output;

pred_prob = output;


%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
activation = exp(non_sigmoid_act);
normalizer = sum(activation, 1);
normalizer = repmat(normalizer, 10 , 1);
P = activation ./ normalizer;

oneFunction = repmat((1:10)', 1, numData);
expandedLabels = repmat(reshape(labels, 1, []), 10, 1);
oneFunction = oneFunction == expandedLabels;

cost = -sum(sum(oneFunction .* log(P)));
cost_gradient = -(oneFunction - P);

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

activation1 = data;
activation2 = hAct{1}.a;

delta2 = cost_gradient;

gradStack{2}.W = delta2 * activation2';
gradStack{2}.b = sum(delta2, 2) .* 0;

delta1 = (W2' * delta2) .* activation2 .* (1 - activation2);

gradStack{1}.W = delta1 * activation1';
gradStack{1}.b = sum(delta1, 2);


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);

end



