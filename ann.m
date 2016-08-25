% An artificial neural network example using
% multi-layer perceptrons with sigmoidal activation.

clear all;
rng(3);

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

  
% The "input layer"
% -----------------

% This layer does not exist, since the "outgoing" connection weights
% are encoded as part of the first hidden layer and the "input neurons"
% would just act as identities.
  

% The (first and only) hidden layer
% ---------------------------------

N_inputs     = numel(X{1});
N_neurons    = 2;
activation   = @(a) 1./(1+exp(-a)); % logistic function
d_activation = @(a) activation(a).*(1-activation(a));
L{1}         = struct(...
                'theta',  0.05*randn(N_neurons, N_inputs), ...
                'bias',   0.05*randn(N_neurons, 1), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );

            
% The output layer
% ----------------

N_inputs     = size(L{1}.theta, 1); % no. inputs is no. previous outputs
N_neurons    = 1;
activation   = L{1}.sigma;          % using the same activation function
d_activation = L{1}.dsigma;
L{2}         = struct(...
                'theta',  0.05*randn(N_neurons, N_inputs), ...
                'bias',   0.05*randn(N_neurons, 1), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );
            
            
% Test execution of the network
% -----------------------------

disp('Untrained network:');

for i=1:numel(X)     % ... for each training sample ...
    
    x = X{i};
    z = feedforward(L, x);
    
    disp([ '  h( ' mat2str(X{i}) ' ) = ' mat2str(z) ]);
end


% Network training
% ----------------

threshold = 0.003;          % stop iterating when J < threshold
N_epochs = 2e4;             % max. number of training rounds
M_epochs = 3e2;             % min. number of training rounds
eta      = 1.;              % the learning rate
my       = 0.1;             % momentum of the learning rate
N_layers = numel(L);
J        = zeros(1, N_epochs);

% Flat spot elimination helps Gradient Descent in very flat error
% surface areas by suggesting some (fake) gradient to move along.
fse      = 0.1;             % flat spot elimination amount

% We are keeping the previous delta values for momentum descent.
prev_change_theta = cell(N_layers, 1);
prev_change_bias  = cell(N_layers, 1);
for j=1:N_layers
    prev_change_theta{j} = zeros(size(L{j}.theta));
    prev_change_bias{j}  = zeros(size(L{j}.bias));
end

for k=1:N_epochs            % ... for each training epoch ...
    
    % prepare space to store all the training activations
    % required during the backpropagation step
    A = cell(N_layers, 1);
    Z = cell(N_layers, 1);
    for j=1:N_layers
        A{j} = nan(1, size(L{j}.theta, 1));
        Out{j} = nan(1, size(L{j}.theta, 1));
    end
    
    % prepare the accumulated delta values
    change_theta = cell(N_layers, 1);
    change_bias  = cell(N_layers, 1);
    for j=1:N_layers
        change_theta{j} = zeros(size(L{j}.theta));
        change_bias{j}  = zeros(size(L{j}.bias));
    end
    
    % running batches of 10 (random) examples each;
    % this makes this approach a Stochastic Gradient Descent.
    range = randi(numel(X), 1, 20);
    
    for i=range             % for each selected training sample
        z = X{i};
        for j=1:N_layers
            weights = L{j}.theta;
            bias    = L{j}.bias;
            sigma   = L{j}.sigma;
            
            a = weights * z + bias;                      
            z = sigma( a );
            
            A{j} = a;       % store the activation for later
            Z{j} = z;       % store the output for later
        end

        e    = Y{i} - Z{N_layers};
        J(k) = J(k) + 0.5 * sum( e.^2 );       % cost function to minimize
        
        % prepare the delta values
        delta = cell(N_layers, 1);             % delta for the weights

        % calculate the error delta of the output layer
        % over all trainings examples (using the dot product)
        output_layer      = L{N_layers};
        gradient          = output_layer.dsigma( A{N_layers} ) + fse;
        delta{N_layers}   = e * gradient;

        % accumulate deltas over the complete batch.
        % The weight change is given as 
        %   dw = delta * input
        % where the input is the output of the previous layer.
        layer_input = Z{N_layers-1};
        change_theta{N_layers} = change_theta{N_layers} ...
                               + delta{N_layers} * layer_input';
        change_bias{N_layers} = change_bias{N_layers} ...
                               + delta{N_layers} * 1;

        % calculate the error delta for all hidden layers
        for j=N_layers-1:-1:1
            current_layer = L{j};
            next_layer    = L{j+1};
            
            % the inputs in the following neurons are affected through
            % our activation function; we thus use its gradient to
            % determine the influence on the error created by each
            % connection weights to connected units.
            delta_k    = delta{j+1};
            theta_k    = next_layer.theta;
            gradient   = current_layer.dsigma( A{j} ) + fse;
            
            delta{j}   = (theta_k'  * delta_k) .* gradient;
            
            % accumulate deltas over the complete batch;
            % the input of the first layer is the actual training input.
            layer_input = X{i};
            if j > 1
                layer_input = Z{j-1};
            end
            change_theta{j} = change_theta{j} + delta{j} * layer_input';
            change_bias{j}  = change_bias{j}  + delta{j} * 1;
        end
        
    end % for each training sample

    % update the hidden and output layer weights
    for j=1:N_layers          
        L{j}.theta = L{j}.theta + eta * change_theta{j} ...
                                +  my * prev_change_theta{j};
        L{j}.bias  = L{j}.bias  + eta * change_bias{j} ...
                                +  my * prev_change_bias{j};
    end
    
    % keeping the current values for momentum-based descent
    prev_change_theta = change_theta;
    prev_change_bias = change_bias;
    
    % adjust the cost for all samples
    J(k) = J(k) / numel(range);
    
    % assume J(theta) is still good here; early exit the process
    if (k >= M_epochs) && (J(k) <= threshold)
        J = J(1:k); % trim away unused slots
        break;
    end
end


% Test execution of the network
% -----------------------------

disp('Trained network:');

for i=1:numel(X)
    z = X{i};

    for j=1:numel(L)
        weights = L{j}.theta;
        bias    = L{j}.bias;
        sigma   = L{j}.sigma;
        
        a = weights*z + bias;
        z = sigma(a);
    end
    
    disp([ '  h( ' mat2str(X{i}) ' ) = ' mat2str(z) ]);
end

close all;
figure;
plot(J); hold on;
ylabel('J(\theta)'); xlabel('Generation');
ylim([0 max(J)]);
xlim([0 numel(J)]);
