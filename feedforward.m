function [ z ] = feedforward( L, x )
%FEEDFORWARD Runs the input through the given network.
    
    % We feed each network layer output as input into the next layer.
    % To initialize the algorithm, we pretend that there's an
    % additional "input layer", whose output is x.
    z = x;

    % Loop through all layers
    for j=1:numel(L)
        layer    = L{j};
        
        weights = layer.theta;
        bias    = layer.bias;
        sigma   = layer.sigma;
        
        a = weights * z + bias;
        z = sigma(a);
    end

end

