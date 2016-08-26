function [ weight_changes, deltas ] = backpropagate( network, feedforward_results, network_error, varargin )
%BACKPROPAGATE Performs the error backpropagation

    p = inputParser;
    addRequired(p, 'network');
    addRequired(p, 'results');
    addRequired(p, 'error');
    addParameter(p, 'fse', 0);
    parse(p, network, feedforward_results, network_error, varargin{:});
    
    network             = p.Results.network;
    feedforward_results = p.Results.results;
    network_error       = p.Results.error;
    fse                 = p.Results.fse;
    
    % prepare the error deltas and gradients
    deltas          = cell(numel(network), 1);
    weight_changes  = cell(numel(network), 1);

    % calculate the error gradient on the network's output layer
    network_results = feedforward_results{end};
    deltas{end}     = network_error;                                                   
    
    % the weight change on the network output
    weight_changes{end} = network_error * network_results.input';          % TODO: explain why

    % perform the actual backpropagation
    for j = numel(network)-1 : -1 : 1

        % obtain the results of the forward propagation
        result  = feedforward_results{j};
        layer   = result.layer;
        net     = result.net;                                              % TODO: find a better name here
        input   = result.input;                                            % TODO: find a better name here

        % obtain the results of the following layer (i.e. downstream)
        next_result  = feedforward_results{j+1};
        next_layer   = next_result.layer;
        next_weights = next_layer.theta(:, 2:end);            % NOTE! removing the bias!
        next_delta   = deltas{j+1};                                        % TODO: explain why
        
        activation_gradient = layer.dsigma( net ) + fse;
        delta   = (next_weights' * next_delta) .* activation_gradient;     % TODO: explain, e.g. http://stats.stackexchange.com/a/130605/26843

        % collect the delta for backpropagation 
        % to the preceding layer
        deltas{j} = delta;

        % calculate the weight change
        weight_changes{j} = delta * input';

        clear e result layer net input;
        clear downstream_result downstream_layer downstream_weights;
        clear activation_gradient delta;
    end % for each layer

end
