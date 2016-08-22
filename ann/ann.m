% An artificial neural network example using
% regular multi-layer perceptrons.

rng(42);

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
% just act as identities.
  
% The (first and only) hidden layer
% ---------------------------------

N_inputs     = numel(X{1});
N_neurons    = 2;
activation   = @(z) 1./(1+exp(-z)); % logistic function
d_activation = @(z) activation(z).*(1-activation(z));
L{1}         = struct(...
                'theta',  randn(N_neurons, N_inputs), ...
                'bias',   randn(N_neurons, 1), ...
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
                'theta',  randn(N_neurons, N_inputs), ...
                'bias',   randn(N_neurons, 1), ...
                'sigma',  activation, ...
                'dsigma', d_activation ...
                );
            
% Test execution of the network
% -----------------------------

disp('Untrained network:');

for i=1:numel(X)     % ... for each training sample ...
    a = X{i};        % The first layer's activation is the network input.

    for j=1:numel(L) % ... for each network layer sample ...
        weights = L{j}.theta;
        bias    = L{j}.bias;
        sigma   = L{j}.sigma;
        
        z = weights*a + bias;
        a = sigma(z);
    end
    
    disp([ '  h( ' mat2str(X{i}) ' ) = ' mat2str(a) ]);
end
