% An artificial neural network example using
% multi-layer perceptrons with sigmoidal activation.

clear all;
%rng(3);

% Learning XOR
% ------------

X = { [0; 0]; 
      [1; 0]; 
      [0; 1]; 
      [1; 1] };
  
Y = { 0;
      1;
      1;
      0 };

assert( numel(X) == numel(Y) );


% The cost function
% -----------------

J  = @(e) (1/2) * (1/numel(e)) * sum(e.^2);     % sum of squared errors
dJ = @(e) (1/numel(e)) * sum(e);                % derivative of J

  
% The "input layer"
% -----------------

% This layer does not exist, since the "outgoing" connection weights
% are encoded as part of the first hidden layer and the "input neurons"
% would just act as identities.
  

% The first hidden layer
% ----------------------
% The first column in the weight matrix theta consists of the bias
% weights to be applied.

N_inputs     = numel(X{1});
N_neurons    = 2;
activation   = @(z) 1./(1+exp(-z)); % logistic function
d_activation = @(z) activation(z).*(1-activation(z));
%activation   = @(z) max(0, z);     % ReLU
%d_activation = @(z) double(z > 0);
L{1}         = struct(...
                'theta',  0.05*randn(N_neurons, 1 + N_inputs), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );

            
% The output layer
% ----------------
% The first column in the weight matrix theta consists of the bias
% weights to be applied.

N_inputs     = size(L{1}.theta, 1); % no. inputs is no. previous outputs
N_neurons    = 1;
activation   = L{1}.sigma;          % using the same activation function
d_activation = L{1}.dsigma;
L{2}         = struct(...
                'theta',  0.05*randn(N_neurons, 1 + N_inputs), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );
            
            
% Test execution of the network
% -----------------------------

disp('Untrained network:');

for t=1:numel(X)     % ... for each training sample ...
    
    x = X{t};
    a = feedforward(L, x);
    
    disp([ '  h( ' mat2str(X{t}) ' ) = ' mat2str(a) ]);
end


% Network training
% ----------------

threshold    = 0.003;           % stop iterating when J < threshold
N_epochs_max = 2e4;             % max. number of training rounds
N_epochs_min = 3e2;             % min. number of training rounds

eta          = 1.0;             % the learning rate
my           = 0.3;             % momentum of the learning rate

N_layers     = numel(L);

% Flat spot elimination helps Gradient Descent in very flat error
% surface areas by suggesting some (fake) gradient to move along.
fse          = 0.1;            % flat spot elimination amount

% We are keeping the previous delta values for momentum descent.
previous_weight_changes = cell(N_layers, 1);
for j=1:N_layers
    previous_weight_changes{j} = zeros(size(L{j}.theta));
end

% track the costs for evaluation of the learning curve
costs = nan(N_epochs_max, 1);


for k=1:N_epochs_max


    % running batches of Q randomly chosen examples (per epoch);
    % this makes this approach a Stochastic Gradient Descent.
    Q = 20;
    range = randi(numel(X), 1, Q);
    %range = mod(k-1, numel(X))+1;

    % for batch training, we need to gather the training results
    % in order to process them collectively.
    training_results = cell(numel(range), 1);

    for t = 1:numel(range) % ... for each training sample

        % pick a training example
        x = X{range(t)}; % network inputs of the current example
        y = Y{range(t)}; % ground truth of the network output (i.e. expected result)

        % perform a feedforward pass but keep around the information
        % about layer inputs and activations.
        [a, results] = feedforward_for_training(L, x);

        % determine the network's error on the current example
        e = a - y;

        % prepare the error deltas and gradients
        deltas          = cell(numel(L), 1);
        weight_changes  = cell(numel(L), 1);

        % calculate the error gradient on the network's output layer
        network_results = results{end};
        deltas{end}     = e;                                                   
        weight_changes{end} = e * network_results.input';

        clear network_results;

        % evaluate the cost function
        cost = J(e);
        assert(isfinite(j));

        % perform the actual backpropagation
        for j = numel(L)-1 : -1 : 1

            % obtain the current layer's error;
            % since we start with the output layer, this is the network error.
            e       = deltas{j+1};

            % obtain the results of the forward propagation
            result  = results{j};
            layer   = result.layer;
            net     = result.net;                                              % TODO: find a better name here
            input   = result.input;                                            % TODO: find a better name here

            % obtain the results of the following layer
            downstream_result  = results{j+1};
            downstream_layer   = downstream_result.layer;
            downstream_weights = downstream_layer.theta(:, 2:end);             % NOTE! removing the bias!

            activation_gradient = layer.dsigma( net ) + fse;
            delta   = (downstream_weights' * e) .* activation_gradient;        % TODO: explain, e.g. http://stats.stackexchange.com/a/130605/26843

            % collect the delta for backpropagation 
            % to the preceding layer
            deltas{j} = delta;

            % calculate the weight change
            weight_changes{j} = delta * input';

            clear e result layer net input;
            clear downstream_result downstream_layer downstream_weights;
            clear activation_gradient delta;
        end

        % deltas are only required for backpropagation
        clear deltas;

        training_results{t} = struct( ...
            'cost', cost, ...
            'weight_changes', {weight_changes} ...     % capital-letter Delta
            );

        clear cost weight_changes;
    
    end % for each training example


    % accumulate the total cost, as well as the combined
    % weight changes over all training examples
    % --------------------------------------------------

    cost = 0;
    weight_changes = cell(numel(L), 1);

    % initialize the cumulative weight changes to zero
    for t=1:numel(L)
        weight_changes{t} = zeros(size(L{t}.theta));
    end

    for t=1:numel(training_results)

        % for each layer, accumulate the weight change
        for w=1:numel(L)
            weight_changes{w} = weight_changes{w} + training_results{t}.weight_changes{w};
        end

        % also sum the costs
        cost = cost + training_results{t}.cost;

    end

    % normalize the cost
    cost     = cost / numel(training_results);
    costs(k) = cost;
    
    
    % normalize the gradients
    for w=1:numel(L)
        weight_changes{w} = weight_changes{w} / numel(training_results);
    end

    % check the change in cost and terminate if it doesn't move
    if k > N_epochs_min && cost < costs(k-1) && (costs(k-1) - cost) < 1E-6
        break;
    end

    
    % Gradient Descent
    % ----------------

    for l=1:numel(L)
        L{l}.theta = L{l}.theta ...
            - eta * weight_changes{l} ...
            -  my * previous_weight_changes{l};
        
        % Store the delta as the previous delta for momentum descent
        previous_weight_changes{l} = weight_changes{l};
    end

end % for k epochs



% Test execution of the network
% -----------------------------

disp('Trained network:');

for t=1:numel(X)     % ... for each training sample ...
    
    x = X{t};
    a = feedforward(L, x);
    
    disp([ '  h( ' mat2str(X{t}) ' ) = ' mat2str(a) ]);
end

close all; figure;
plot(costs);
xlabel('Generation');
ylabel('J(\theta)');