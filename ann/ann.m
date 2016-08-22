% An artificial neural network example using
% regular multi-layer perceptrons.

rng(42);

% Learning XOR

X = { [0 0]; 
      [1 0]; 
      [0 1]; 
      [1 1] };
  
Y = { 0;
      1;
      1;
      0 };

% The "input layer"
% This layer does not exist, since the "outgoing" connection weights
% are encoded as part of the first hidden layer and the "input neurons"
% just act as identities.
  
% Defining the hidden layer

N_inputs     = numel(X{1});
N_neurons    = 2;
activation   = @(z) 1/(1+exp(-z));
d_activation = @(z) activation(z)*(1-activation(z));
L{1}         = struct(...
                'theta',  randn(N_neurons, N_inputs), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );

% Defining the output layer

N_inputs     = size(L{1}.theta, 1);
N_neurons    = 1;
activation   = @(z) 1/(1+exp(-z));
d_activation = @(z) activation(z)*(1-activation(z));
L{2}         = struct(...
                'theta',  randn(N_neurons, N_inputs), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );
