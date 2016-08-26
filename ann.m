% An artificial neural network example using
% multi-layer perceptrons with sigmoidal activation.

clear all;
%rng(3);

% Learning Boolean functions
% --------------------------

% 0 0 = OR
% 0 1 = AND
% 1 0 = XOR
% 1 1 = NAND

X = { % XOR examples
      [1; 0;    0; 0]; 
      [1; 0;    1; 0]; 
      [1; 0;    0; 1]; 
      [1; 0;    1; 1];
      % OR examples
      [0; 0;    0; 0]; 
      [0; 0;    1; 0]; 
      [0; 0;    0; 1]; 
      [0; 0;    1; 1];
      % AND examples
      [0; 1;    0; 0]; 
      [0; 1;    1; 0]; 
      [0; 1;    0; 1]; 
      [0; 1;    1; 1];
      % NAND examples
      [1; 1;    0; 0]; 
      [1; 1;    1; 0]; 
      [1; 1;    0; 1]; 
      [1; 1;    1; 1] };
  
Y = { % XOR examples
      0;
      1;
      1;
      0;
      % OR examples
      0;
      1;
      1;
      1;
      % AND examples
      0;
      0;
      0;
      1;
      % NAND examples
      1;
      1;
      1;
      0 };

assert( numel(X) == numel(Y) );


% The cost function
% -----------------

J  = @(e) (1/numel(e)) * (1/2) * sum(e.^2);     % sum of squared errors
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
N_outputs    = 4;
activation   = @(z) 1./(1+exp(-z)); % logistic function
d_activation = @(z) activation(z).*(1-activation(z));
%activation   = @(z) max(0, z);     % ReLU
%d_activation = @(z) double(z > 0);
L{1}         = struct(...
                'theta',  0.05*randn(N_outputs, 1 + N_inputs), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );
            
            
% The output layer
% ----------------
% The first column in the weight matrix theta consists of the bias
% weights to be applied.

N_inputs     = size(L{1}.theta, 1); % no. inputs is no. previous outputs
N_outputs    = 1;
activation   = L{1}.sigma;          % using the same activation function
d_activation = L{1}.dsigma;
L{2}         = struct(...
                'theta',  0.05*randn(N_outputs, 1 + N_inputs), ...
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

N_layers     = numel(L);

threshold    = 1E-5;            % stop iterating when delta J < threshold
N_epochs_max = 2e4;             % max. number of training rounds
N_epochs_min = 3e3;             % min. number of training rounds

% Select the gradient descent algorithm
%gradient_descent = momentum_gradient_descent( ...
%                    L, ...
%                    'learning_rate', 1.0, ...
%                    'momentum', 0.1);

gradient_descent = accelerated_gradient_descent( ...
                    L, ...
                    'acceleration', 1.5);
                
% Flat spot elimination helps Gradient Descent in very flat error
% surface areas by suggesting some (fake) gradient to move along.
fse          = 0.1;            % flat spot elimination amount

% track the costs for evaluation of the learning curve
costs = nan(N_epochs_max, 1);

% execute the training epochs
total_duration = tic;
tic;
for k=1:N_epochs_max


    % running batches of Q randomly chosen examples (per epoch);
    % this makes this approach a Stochastic Gradient Descent.
    Q = 10;
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
        
        % evaluate the cost function
        cost = J(e);
        assert(isfinite(cost));
        
        % perform the error backpropagation
        [weight_changes, ~] = backpropagate(dJ, L, results, e, 'fse', fse);       
        training_results{t} = struct( ...
            'cost', cost, ...
            'weight_changes', {weight_changes} ...                         % TODO: capital-letter Delta
            );

        clear cost weight_changes;
            
    end % for each training example


    % accumulate the total cost, as well as the combined
    % weight changes over all training examples
    % --------------------------------------------------

    % sum all the costs an normalize by number of training results
    N_results = numel(training_results);
    costs(k) = sum( cellfun(@(r) r.cost, training_results) ) / N_results;
    
    % initialize the cumulative weight changes to zero
    weight_changes = cellfun( ...
        @(L) zeros( size(L.theta) ), ...
        L, ...
        'UniformOutput', false);

    % sum each layer's weight changes over all training results
    for t=1:numel(training_results)
        result = training_results{t};
        for w=1:numel(L)
            weight_changes{w} = weight_changes{w} + result.weight_changes{w};
        end
    end

    % normalize the weight changes over all results
    weight_changes = cellfun(...
        @(w) w / N_results, ...
        weight_changes, ...
        'UniformOutput', false);
    
    % check the change in cost and terminate if it doesn't move
    if (k > N_epochs_min) && ...
       (costs(k) < costs(k-1)) && ...
       ((costs(k-1) - costs(k)) < threshold)
   
        disp('Cost change less than threshold; aborting.');
        break;
    end

    % waiting aid
    elapsed = toc;
    if elapsed >= 5
        tic;
        disp(['duration ' num2str(toc(total_duration)) 's, ' ...
              'gen ' num2str(k) ', ' ...
              'J(theta) = ' num2str(costs(k))]);
    end
    
    
    % Gradient Descent
    % ----------------

    L = gradient_descent(L, weight_changes);

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