function [ a ] = feedforward( network, input )
%FEEDFORWARD Runs the input through the given network.
    
    % We feed each network layer output as input into the next layer.
    % To initialize the algorithm, we pretend that there's an
    % additional "input layer", whose output is x.
    a = input;

    % Loop through all layers
    for j=1:numel(network)
        layer   = network{j};
        
        weights = layer.theta;
        sigma   = layer.sigma;
        
        z = weights * [1; a]; % appending 1 for the bias activation
        a = sigma(z);
    end

end

